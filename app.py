from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import torch

app = Flask(__name__, template_folder="templates")
CORS(app)

SHP_DATA = []
SHP_INDEX_BY_ID = {}

# Base model and tokenizer (shared across all LoRA adapters)
BASE_MODEL = None
BASE_TOKENIZER = None
CURRENT_BACKBONE = None
REWARD_MODEL = None

# Available backbones (can be extended)
AVAILABLE_BACKBONES = [
    "google/gemma-3-4b-it",
    "google/gemma-2-9b-it",
    # Add more as needed
]


def load_shp_data():
    """Load SHP dataset using src_utils."""
    global SHP_DATA, SHP_INDEX_BY_ID
    SHP_DATA = []
    SHP_INDEX_BY_ID = {}

    from src_utils import load_shp_dataset
    
    print("[SHP] Loading Stanford Human Preferences dataset...")
    dataset = load_shp_dataset(
        split="train",
        max_samples=1000,
        seed=42,
    )
    
    SHP_DATA = list(dataset)
    
    # Use index as ID
    for idx, item in enumerate(SHP_DATA):
        item_id = str(idx)
        SHP_INDEX_BY_ID[item_id] = item
    
    print(f"[SHP] loaded {len(SHP_DATA)} examples")


def get_lora_paths(backbone_name: str):
    """Get paths to trained LoRA adapters for a given backbone.
    Tries multiple possible paths:
    1. checkpoints_lora_dpo/{model_dir}/{algorithm}-lora
    2. checkpoints/{model_name}/{algorithm}-lora
    """
    model_dir = backbone_name.replace("/", "-")
    
    # Try checkpoints_lora_dpo first (as shown in project layout)
    paths_v1 = {
        "sft": os.path.join("checkpoints_lora_dpo", model_dir, "sft-lora"),
        "dpo": os.path.join("checkpoints_lora_dpo", model_dir, "dpo-lora"),
        "ppo": os.path.join("checkpoints_lora_dpo", model_dir, "ppo-lora"),
        "grpo": os.path.join("checkpoints_lora_dpo", model_dir, "grpo-lora"),
        "rm": os.path.join("checkpoints_lora_dpo", model_dir, "rm-lora"),
    }
    
    # Try checkpoints/{model_name} format (as in train.py)
    paths_v2 = {
        "sft": os.path.join("checkpoints", backbone_name, "sft-lora"),
        "dpo": os.path.join("checkpoints", backbone_name, "dpo-lora"),
        "ppo": os.path.join("checkpoints", backbone_name, "ppo-lora"),
        "grpo": os.path.join("checkpoints", backbone_name, "grpo-lora"),
        "rm": os.path.join("checkpoints", backbone_name, "rm-lora"),
    }
    
    # Return the path that exists, or v1 as default
    result = {}
    for key in ["sft", "dpo", "ppo", "grpo", "rm"]:
        if os.path.exists(paths_v1[key]):
            result[key] = paths_v1[key]
        elif os.path.exists(paths_v2[key]):
            result[key] = paths_v2[key]
        else:
            # Default to v1 format (even if doesn't exist)
            result[key] = paths_v1[key]
    
    return result


def load_base_model(backbone_name: str):
    """Load base model and tokenizer (shared across all LoRA adapters)."""
    global BASE_MODEL, BASE_TOKENIZER, CURRENT_BACKBONE
    
    # If already loaded with the same backbone, no need to reload
    if BASE_MODEL is not None and CURRENT_BACKBONE == backbone_name:
        return
    
    # If different backbone, need to reload
    if BASE_MODEL is not None and CURRENT_BACKBONE != backbone_name:
        # Clear old model from memory
        del BASE_MODEL
        del BASE_TOKENIZER
        BASE_MODEL = None
        BASE_TOKENIZER = None
        # Also clear reward model
        global REWARD_MODEL
        if REWARD_MODEL is not None:
            del REWARD_MODEL
            REWARD_MODEL = None
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        from src_utils import load_model
        
        print(f"[Base Model] Loading backbone: {backbone_name}...")
        BASE_MODEL, BASE_TOKENIZER = load_model(backbone_name)
        BASE_MODEL.eval()
        CURRENT_BACKBONE = backbone_name
        print(f"[Base Model] ✅ Backbone loaded successfully")
    except Exception as e:
        print(f"[Base Model] ❌ Error loading backbone: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_lora_adapter(backbone_name: str, adapter_type: str):
    """Load a LoRA adapter on top of the base model.
    
    Reuses the already-loaded BASE_MODEL to avoid reloading the backbone.
    Only loads the LoRA adapter weights.
    """
    from peft import PeftModel
    
    if BASE_MODEL is None or CURRENT_BACKBONE != backbone_name:
        load_base_model(backbone_name)
    
    lora_paths = get_lora_paths(backbone_name)
    adapter_path = lora_paths.get(adapter_type)
    
    if not adapter_path or not os.path.exists(adapter_path):
        return None, f"LoRA adapter not found at {adapter_path}"
    
    try:
        # Reuse the already-loaded base model
        # PeftModel.from_pretrained will load only the adapter weights
        # Note: We need to ensure BASE_MODEL is not already a PeftModel
        base_model_to_use = BASE_MODEL
        
        # If BASE_MODEL is already wrapped in PeftModel, we need the underlying model
        # But since we load it fresh in load_base_model, it should be the raw model
        if hasattr(BASE_MODEL, 'get_base_model'):
            # It's already a PeftModel, get the base
            base_model_to_use = BASE_MODEL.get_base_model()
        elif hasattr(BASE_MODEL, 'base_model') and hasattr(BASE_MODEL.base_model, 'model'):
            # Alternative access pattern
            base_model_to_use = BASE_MODEL.base_model.model
        
        # Load adapter on the base model
        # This only loads the LoRA adapter weights, not the full model
        model_with_adapter = PeftModel.from_pretrained(
            base_model_to_use,
            adapter_path,
            is_trainable=False,
        )
        model_with_adapter.eval()
        
        return model_with_adapter, None
    except Exception as e:
        # If direct PeftModel loading fails, fall back to prepare_lora approach
        # but still reuse the base model
        try:
            from src_lora import prepare_lora
            
            # Create a shallow copy of the model structure to avoid state conflicts
            # This is still more efficient than reloading from disk
            import copy
            base_model_copy = copy.deepcopy(BASE_MODEL)
            base_model_copy.eval()
            
            model_with_adapter = prepare_lora(
                base_model_copy,
                checkpoint_path=adapter_path
            )
            model_with_adapter.eval()
            
            for param in model_with_adapter.parameters():
                param.requires_grad = False
            
            return model_with_adapter, None
        except Exception as e2:
            import traceback
            traceback.print_exc()
            return None, f"Failed to load adapter: {str(e)}, fallback also failed: {str(e2)}"


def load_reward_model_for_backbone(backbone_name: str):
    """Load reward model for the current backbone."""
    global REWARD_MODEL
    
    if BASE_MODEL is None or CURRENT_BACKBONE != backbone_name:
        load_base_model(backbone_name)
    
    # If already loaded, return
    if REWARD_MODEL is not None:
        return None
    
    lora_paths = get_lora_paths(backbone_name)
    rm_path = lora_paths.get("rm")
    
    if not rm_path or not os.path.exists(rm_path):
        return f"Reward model not found at {rm_path}"
    
    try:
        from src_rm import load_reward_model
        
        print(f"[Reward Model] Loading reward model from {rm_path}...")
        # Reward model uses same backbone, optionally with SFT LoRA
        # Check if SFT LoRA exists for the reward model
        sft_path = lora_paths.get("sft")
        sft_lora_path = sft_path if sft_path and os.path.exists(sft_path) else None
        
        REWARD_MODEL, _ = load_reward_model(
            backbone_name,
            rm_path,
            sft_lora_path=sft_lora_path,
        )
        REWARD_MODEL.eval()
        
        # Debug: verify reward head is loaded (not randomly initialized)
        if hasattr(REWARD_MODEL, 'reward_head'):
            reward_head_params = list(REWARD_MODEL.reward_head.parameters())
            if len(reward_head_params) > 0:
                param_sum = sum(p.sum().item() for p in reward_head_params)
                param_norm = sum(p.norm().item() for p in reward_head_params)
                print(f"[Reward Model Debug] Reward head param sum: {param_sum:.6f}, norm: {param_norm:.6f}")
                
                # Check if reward head is zero or near-zero (might indicate not loaded properly)
                if abs(param_sum) < 1e-6:
                    print(f"[Reward Model Warning] Reward head parameters are near-zero! Model might not be loaded correctly.")
        
        print(f"[Reward Model] ✅ Reward model loaded successfully")
        return None
    except Exception as e:
        print(f"[Reward Model] ❌ Error loading reward model: {e}")
        import traceback
        traceback.print_exc()
        return str(e)


def compute_reward_for_response(prompt: str, response: str):
    """Compute reward for a prompt-response pair."""
    if REWARD_MODEL is None:
        return None, "Reward model not loaded"
    
    try:
        from src_rm import compute_reward
        
        reward = compute_reward(
            REWARD_MODEL,
            BASE_TOKENIZER,
            prompt,
            response,
            max_length=512,
        )
        return reward, None
    except Exception as e:
        return None, str(e)


def generate_with_lora(backbone_name: str, adapter_type: str, prompt: str, max_new_tokens: int = 128):
    """Generate response using backbone + LoRA adapter."""
    if BASE_TOKENIZER is None or CURRENT_BACKBONE != backbone_name:
        load_base_model(backbone_name)
    
    # Load adapter dynamically
    model_with_adapter, error = load_lora_adapter(backbone_name, adapter_type)
    if error:
        return None, error
    
    try:
        from src_utils import generate_response
        
        with torch.no_grad():
            response = generate_response(
                model_with_adapter,
                BASE_TOKENIZER,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Clean up adapter from memory (optional, but helps with memory)
        # The adapter will be reloaded next time if needed
        del model_with_adapter
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return response, None
    except Exception as e:
        return None, str(e)




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/index/search")
def api_shp_search():
    """Right panel search: simple keyword search on prompt / chosen / rejected."""
    q = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 20))

    if not SHP_DATA:
        return jsonify({"results": [], "total": 0})

    if not q:
        # Return items with index as ID
        items = [(str(idx), item) for idx, item in enumerate(SHP_DATA[:limit])]
    else:
        items = []
        for idx, s in enumerate(SHP_DATA):
            if len(items) >= limit:
                break
            text = " ".join(
                [
                    str(idx),
                    s.get("prompt", ""),
                    s.get("chosen", ""),
                    s.get("rejected", ""),
                ]
            ).lower()
            if q in text:
                items.append((str(idx), s))

    out = [
        {
            "id": item_id,
            "question": item.get("prompt", ""),
            "answer": item.get("chosen", ""),
        }
        for item_id, item in items
    ]
    return jsonify({"results": out, "total": len(SHP_DATA)})


@app.route("/api/index/<string:item_id>")
def api_shp_item(item_id: str):
    """Left panel details for a specific SHP example."""
    sample = SHP_INDEX_BY_ID.get(item_id)
    if not sample:
        return jsonify({"error": "not found"}), 404

    return jsonify(
        {
            "id": item_id,
            "question": sample.get("prompt", ""),
            "answer": sample.get("chosen", ""),
            "rejected": sample.get("rejected", ""),
            "answer_aliases": [],
            "q_decomposition": [],
            "paragraphs": [],
            "graph": {
                "nodes": [],
                "edges": [],
            },
        }
    )


@app.route("/api/backbones")
def api_get_backbones():
    """Get list of available backbones."""
    return jsonify({"backbones": AVAILABLE_BACKBONES})


@app.route("/api/backbones/<path:backbone_name>/lora-paths")
def api_get_lora_paths(backbone_name: str):
    """Get available LoRA adapter paths for a backbone."""
    paths = get_lora_paths(backbone_name)
    available = {}
    for adapter_type, path in paths.items():
        available[adapter_type] = {
            "path": path,
            "exists": os.path.exists(path),
        }
    return jsonify({"backbone": backbone_name, "adapters": available})


@app.route("/api/models/load", methods=["POST"])
def api_load_models():
    """Load backbone model and prepare for inference."""
    data = request.get_json()
    backbone_name = data.get("backbone") if data else None
    
    if not backbone_name:
        return jsonify({"error": "backbone not specified"}), 400
    
    if backbone_name not in AVAILABLE_BACKBONES:
        return jsonify({"error": f"backbone {backbone_name} not available"}), 400
    
    try:
        load_base_model(backbone_name)
        
        # Check which adapters are available
        lora_paths = get_lora_paths(backbone_name)
        adapters_status = {}
        for adapter_type, path in lora_paths.items():
            adapters_status[adapter_type] = {
                "path": path,
                "exists": os.path.exists(path),
            }
        
        # Try to load reward model if available
        reward_model_error = None
        reward_model_loaded = False
        if REWARD_MODEL is None:
            reward_model_error = load_reward_model_for_backbone(backbone_name)
            reward_model_loaded = REWARD_MODEL is not None
        else:
            reward_model_loaded = True
        
        # Add reward model status to adapters
        if "rm" in adapters_status:
            adapters_status["rm"]["loaded"] = reward_model_loaded
            if reward_model_error:
                adapters_status["rm"]["error"] = reward_model_error
        
        return jsonify({
            "success": True,
            "backbone": backbone_name,
            "message": f"Backbone {backbone_name} loaded successfully",
            "adapters": adapters_status,
            "reward_model_loaded": reward_model_loaded,
            "reward_model_error": reward_model_error if reward_model_error else None,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route("/api/models/status")
def api_get_model_status():
    """Get current model loading status."""
    return jsonify({
        "loaded": BASE_MODEL is not None,
        "backbone": CURRENT_BACKBONE,
        "tokenizer_loaded": BASE_TOKENIZER is not None,
    })


@app.route("/api/index/<string:item_id>/generate/<string:adapter_type>")
def api_shp_generate_single(item_id: str, adapter_type: str):
    """Generate response for a single adapter type (for real-time streaming)."""
    global REWARD_MODEL, BASE_MODEL, BASE_TOKENIZER, CURRENT_BACKBONE
    
    sample = SHP_INDEX_BY_ID.get(item_id)
    if not sample:
        return jsonify({"error": "not found"}), 404
    
    prompt = sample.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt not found"}), 400
    
    # Check if model is loaded
    if BASE_MODEL is None or BASE_TOKENIZER is None:
        return jsonify({"error": "Model not loaded. Please load the model first."}), 400
    
    backbone_name = CURRENT_BACKBONE
    if not backbone_name:
        return jsonify({"error": "No backbone currently loaded"}), 400
    
    # Try to load reward model if not already loaded
    reward_model_error = None
    if REWARD_MODEL is None:
        reward_model_error = load_reward_model_for_backbone(backbone_name)
        if reward_model_error:
            print(f"[Warning] Failed to load reward model: {reward_model_error}")
    
    # Get max_new_tokens from query parameter
    max_new_tokens = int(request.args.get("max_tokens", 64))
    max_new_tokens = max(1, min(512, max_new_tokens))  # Clamp between 1 and 512
    
    if adapter_type not in ["sft", "dpo", "ppo", "grpo"]:
        return jsonify({"error": f"Invalid adapter type: {adapter_type}"}), 400
    
    lora_paths = get_lora_paths(backbone_name)
    adapter_path = lora_paths.get(adapter_type)
    adapter_exists = adapter_path and os.path.exists(adapter_path)
    
    if not adapter_exists:
        return jsonify({
            "adapter_type": adapter_type,
            "response": None,
            "error": f"Adapter not found at {adapter_path}",
            "loaded": False,
            "reward": None,
            "reward_error": None,
        })
    
    response, error = generate_with_lora(backbone_name, adapter_type, prompt, max_new_tokens)
    
    if error:
        return jsonify({
            "adapter_type": adapter_type,
            "response": None,
            "error": error,
            "loaded": False,
            "reward": None,
            "reward_error": None,
        })
    
    # Debug: verify response was generated
    print(f"[Generate Debug] {adapter_type} - Generated response length: {len(response) if response else 0}")
    if response:
        response_preview = response[:150] if len(response) > 150 else response
        print(f"[Generate Debug] {adapter_type} - Response preview: {response_preview}...")
    
    # Compute reward if reward model is loaded
    reward = None
    reward_error = None
    if REWARD_MODEL is not None:
        try:
            print(f"[Reward Debug] {adapter_type} - Computing reward with prompt len: {len(prompt)}, response len: {len(response) if response else 0}")
            reward, reward_error = compute_reward_for_response(prompt, response)
            print(f"[Reward] Computed reward for {adapter_type}: {reward}, error: {reward_error}")
        except Exception as e:
            print(f"[Reward] Error computing reward for {adapter_type}: {e}")
            import traceback
            traceback.print_exc()
            reward_error = str(e)
    else:
        reward_error = reward_model_error or "Reward model not loaded"
        print(f"[Reward] Reward model not loaded for {adapter_type}: {reward_error}")
    
    return jsonify({
        "adapter_type": adapter_type,
        "response": response,
        "error": None,
        "loaded": True,
        "reward": reward,
        "reward_error": reward_error,
    })


if __name__ == "__main__":
    load_shp_data()
    # Models will be loaded lazily on first request
    app.run(host="0.0.0.0", port=5002, debug=True)


