import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src_utils import load_shp_dataset, load_model, format_chat
from src_lora import prepare_lora


def get_hidden_size(model):
    """
    Get the hidden size from a model, handling different architectures.
    
    Tries multiple common attribute names:
    - hidden_size (BERT, RoBERTa, etc.)
    - d_model (T5, etc.)
    - n_embd (GPT models)
    - dim (some models)
    
    Falls back to getting it from the model's embedding layer or first layer.
    """
    config = model.config
    
    # First, try to get from embedding layer (most reliable)
    # Check common embedding layer locations
    if hasattr(model, 'embed_tokens'):
        return model.embed_tokens.embedding_dim
    elif hasattr(model, 'text_model'):
        # For Gemma3 and similar models with text_model
        if hasattr(model.text_model, 'embed_tokens'):
            return model.text_model.embed_tokens.embedding_dim
        elif hasattr(model.text_model, 'embedding'):
            return model.text_model.embedding.embedding_dim
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            return model.model.embed_tokens.embedding_dim
        elif hasattr(model.model, 'embedding'):
            return model.model.embedding.embedding_dim
    
    # For models with nested configs (e.g., Gemma3 with text_config)
    if hasattr(config, 'text_config'):
        text_config = config.text_config
        if hasattr(text_config, 'hidden_size'):
            return text_config.hidden_size
        elif hasattr(text_config, 'd_model'):
            return text_config.d_model
        elif hasattr(text_config, 'n_embd'):
            return text_config.n_embd
        elif hasattr(text_config, 'dim'):
            return text_config.dim
    
    # Try common attribute names in config
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    elif hasattr(config, 'd_model'):
        return config.d_model
    elif hasattr(config, 'n_embd'):
        return config.n_embd
    elif hasattr(config, 'dim'):
        return config.dim
    elif hasattr(config, 'vocab_size') and hasattr(model, 'lm_head'):
        # Try to infer from lm_head if available
        if hasattr(model.lm_head, 'in_features'):
            return model.lm_head.in_features
        elif hasattr(model.lm_head, 'weight'):
            return model.lm_head.weight.shape[1]
    
    # Last resort: try to infer from first transformer layer
    if hasattr(model, 'layers') and len(model.layers) > 0:
        first_layer = model.layers[0]
        if hasattr(first_layer, 'self_attn'):
            if hasattr(first_layer.self_attn, 'q_proj'):
                return first_layer.self_attn.q_proj.in_features
    elif hasattr(model, 'text_model') and hasattr(model.text_model, 'layers') and len(model.text_model.layers) > 0:
        # For Gemma3 text_model layers
        first_layer = model.text_model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
            return first_layer.self_attn.q_proj.in_features
    elif hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
        first_layer = model.model.layers[0]
        if hasattr(first_layer, 'self_attn'):
            if hasattr(first_layer.self_attn, 'q_proj'):
                return first_layer.self_attn.q_proj.in_features
    
    raise ValueError(
        f"Could not determine hidden size from model. "
        f"Config type: {type(config)}, Config attributes: {[a for a in dir(config) if not a.startswith('_')]}"
    )


class RewardModel(nn.Module):
    """
    Reward model wrapper that adds a reward head to a base language model.
    The reward head takes the final hidden state and outputs a scalar reward.
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Reward head: maps hidden_size -> 1
        hidden_size = get_hidden_size(base_model)
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        
        # Try to move reward_head to the same device and dtype as base_model
        # (works if base_model is already on a device)
        try:
            base_model_param = next(base_model.parameters())
            base_model_device = base_model_param.device
            base_model_dtype = base_model_param.dtype
            self.reward_head = self.reward_head.to(device=base_model_device, dtype=base_model_dtype)
        except (StopIteration, AttributeError):
            # Base model might use device_map="auto" - will handle in forward
            pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass through the reward model.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            rewards: (batch_size,) scalar rewards
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get hidden states: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]
        
        # Extract final hidden state for each sequence
        # Use the last non-padding token (or EOS token if present)
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            # Ensure seq_lengths is on the same device as hidden_states
            seq_lengths = seq_lengths.to(hidden_states.device)
            # Clamp to valid range
            seq_lengths = seq_lengths.clamp(min=0, max=hidden_states.shape[1] - 1)
            # Gather final hidden states: (batch_size, hidden_size)
            batch_indices = torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
            )
            final_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            # If no attention mask, use the last token
            final_hidden = hidden_states[:, -1, :]
        
        # Ensure reward_head is on the same device and dtype as final_hidden
        reward_head_param = next(self.reward_head.parameters())
        reward_head_device = reward_head_param.device
        reward_head_dtype = reward_head_param.dtype
        
        if final_hidden.device != reward_head_device or final_hidden.dtype != reward_head_dtype:
            # Move reward_head to final_hidden device and dtype (CUDA is preferred over CPU)
            if final_hidden.device.type == 'cuda':
                self.reward_head = self.reward_head.to(device=final_hidden.device, dtype=final_hidden.dtype)
            else:
                # If final_hidden is on CPU, move it to reward_head device and dtype
                final_hidden = final_hidden.to(device=reward_head_device, dtype=reward_head_dtype)
        
        # Compute scalar reward: (batch_size,)
        rewards = self.reward_head(final_hidden).squeeze(-1)
        
        return rewards


def build_reward_inputs(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    device: torch.device,
):
    """
    Build inputs for reward model training.
    Formats prompt-response pair using chat template.
    
    Returns:
        input_ids: (1, seq_len)
        attention_mask: (1, seq_len) or None
    """
    # Debug: verify response is not empty
    if not response or len(response.strip()) == 0:
        print(f"[WARNING] build_reward_inputs received empty response! prompt length: {len(prompt)}")
    
    # Format full conversation
    full_text = format_chat(
        tokenizer,
        prompt=prompt,
        response=response,
    )
    
    # Debug: verify response is in formatted text
    response_in_text = response.strip() in full_text if response else False
    if not response_in_text and response:
        print(f"[WARNING] Response not found in formatted text! Response: {response[:50]}...")
    
    # Set padding side to left so that response tokens are at the end
    # This ensures the last non-padding token is from the response
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Debug: print input summary
    # Determine pad token ID (Gemma uses 0, but check tokenizer)
    if tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    else:
        pad_token_id = 0
    
    # Use attention_mask if available (more reliable than checking pad_token_id)
    if attention_mask is not None:
        non_padding_tokens = attention_mask[0].sum().item()
        last_non_pad_pos = non_padding_tokens - 1  # 0-indexed
        # Get actual token IDs from the sequence (non-padding part)
        actual_token_ids = input_ids[0][:non_padding_tokens].cpu().tolist()
    else:
        # Fallback: check pad_token_id
        non_padding_mask = input_ids[0] != pad_token_id
        non_padding_tokens = non_padding_mask.sum().item()
        last_non_pad_pos = (non_padding_mask).nonzero(as_tuple=False)[-1].item() if non_padding_tokens > 0 else -1
        actual_token_ids = input_ids[0][non_padding_mask].cpu().tolist()
    
    # Get last 20 token IDs for debugging
    token_snippet = actual_token_ids[-20:] if len(actual_token_ids) >= 20 else actual_token_ids
    
    # Also check response tokens in the actual sequence
    # Decode last part to verify response is included
    try:
        last_50_tokens_text = tokenizer.decode(actual_token_ids[-50:] if len(actual_token_ids) >= 50 else actual_token_ids, skip_special_tokens=False)
        response_check = response[:50].strip().lower() in last_50_tokens_text.lower() if response else False
    except:
        response_check = False
        last_50_tokens_text = ""
    
    print(f"[Reward Input Debug] Prompt len: {len(prompt)}, Response len: {len(response)}, Full text len: {len(full_text)}")
    print(f"[Reward Input Debug] Non-pad tokens: {non_padding_tokens}, Last non-pad pos: {last_non_pad_pos}, Pad token ID: {pad_token_id}")
    print(f"[Reward Input Debug] Last 20 token IDs: {token_snippet}")
    print(f"[Reward Input Debug] Response in last 50 tokens: {response_check}, Last 50 tokens decoded (preview): {last_50_tokens_text[-100:]}")
    
    return input_ids, attention_mask


def reward_ranking_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
):
    """
    Logistic ranking loss for reward model training.
    
    Loss = -log σ(r_chosen - r_rejected)
    where σ is the sigmoid function.
    
    This encourages chosen responses to have higher rewards than rejected ones.
    
    Args:
        chosen_rewards: (batch_size,) rewards for chosen responses
        rejected_rewards: (batch_size,) rewards for rejected responses
    
    Returns:
        loss: scalar tensor
    """
    # Compute reward difference
    reward_diff = chosen_rewards - rejected_rewards
    
    # Logistic ranking loss: -log σ(diff)
    # Using log_sigmoid for numerical stability
    loss = -F.logsigmoid(reward_diff).mean()
    
    return loss


def concatenated_forward_rewards(
    reward_model: RewardModel,
    batch: dict,
    pad_token_id: int = 0,
):
    """
    Concatenate chosen/rejected along batch dimension
    and run a single forward pass through the reward model.
    Pads sequences to the same length before concatenation.
    
    Args:
        reward_model: RewardModel instance
        batch: dict with chosen/rejected input_ids and attention_masks
        pad_token_id: padding token ID
    
    Returns:
        chosen_rewards: (batch_size,)
        rejected_rewards: (batch_size,)
    """
    ch_input_ids = batch["chosen_input_ids"]
    rj_input_ids = batch["rejected_input_ids"]
    
    # Get max sequence length
    max_len = max(ch_input_ids.shape[1], rj_input_ids.shape[1])
    
    # Pad chosen sequences
    if ch_input_ids.shape[1] < max_len:
        pad_len = max_len - ch_input_ids.shape[1]
        ch_input_ids = F.pad(ch_input_ids, (0, pad_len), value=pad_token_id)
    
    # Pad rejected sequences
    if rj_input_ids.shape[1] < max_len:
        pad_len = max_len - rj_input_ids.shape[1]
        rj_input_ids = F.pad(rj_input_ids, (0, pad_len), value=pad_token_id)
    
    input_ids = torch.cat([ch_input_ids, rj_input_ids], dim=0)
    
    if batch["chosen_attention_mask"] is None:
        attention_mask = None
    else:
        ch_attn = batch["chosen_attention_mask"]
        rj_attn = batch["rejected_attention_mask"]
        
        # Pad attention masks
        if ch_attn.shape[1] < max_len:
            pad_len = max_len - ch_attn.shape[1]
            ch_attn = F.pad(ch_attn, (0, pad_len), value=0)
        if rj_attn.shape[1] < max_len:
            pad_len = max_len - rj_attn.shape[1]
            rj_attn = F.pad(rj_attn, (0, pad_len), value=0)
        
        attention_mask = torch.cat([ch_attn, rj_attn], dim=0)
    
    # Forward pass
    rewards = reward_model(input_ids=input_ids, attention_mask=attention_mask)
    
    bsz = batch["chosen_input_ids"].shape[0]
    return rewards[:bsz], rewards[bsz:]


def run_reward_model_training(
    model_name: str,
    output_dir: str,
    split: str = "train",
    max_samples: int = 1000,
    batch_size: int = 2,
    lr: float = 1e-5,
    max_length: int = 512,
    steps: int | None = None,
    sft_lora_path: str | None = None,
):
    """
    Train a reward model using pairwise preference data.
    
    The reward model is trained to assign higher scalar rewards to chosen
    responses compared to rejected responses using a logistic ranking loss.
    
    Args:
        model_name: base model name (e.g., google/gemma-2-2b-it)
        output_dir: directory to save reward model checkpoint
        split: dataset split
        max_samples: number of training samples
        batch_size: training batch size
        lr: learning rate
        max_length: max token length
        steps: number of training steps (if None, uses all samples)
        sft_lora_path: optional SFT LoRA checkpoint to initialize from
    """
    
    # 1. Load dataset
    dataset = load_shp_dataset(
        split=split,
        max_samples=max_samples,
    )
    
    # 2. Load base model + tokenizer
    model, tokenizer = load_model(model_name)
    
    # Optionally load SFT LoRA adapter
    if sft_lora_path is not None:
        model = prepare_lora(model, checkpoint_path=sft_lora_path)
    
    # Wrap model with reward head
    reward_model = RewardModel(model)
    reward_model.train()
    reward_model.base_model.config.use_cache = False
    
    # 3. Optimizer (only train reward head + optionally base model)
    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=lr,
    )
    
    # 4. Training loop
    loss_history = []
    pbar = tqdm(dataset, desc="Reward Model Training")
    
    batch_chosen_inputs = []
    batch_chosen_masks = []
    batch_rejected_inputs = []
    batch_rejected_masks = []
    
    for step, sample in enumerate(pbar):
        if steps is not None and step >= steps:
            break
        
        # Build chosen/rejected inputs
        ch_input_ids, ch_attn = build_reward_inputs(
            tokenizer,
            sample["prompt"],
            sample["chosen"],
            max_length,
            reward_model.base_model.device,
        )
        
        rj_input_ids, rj_attn = build_reward_inputs(
            tokenizer,
            sample["prompt"],
            sample["rejected"],
            max_length,
            reward_model.base_model.device,
        )
        
        batch_chosen_inputs.append(ch_input_ids.squeeze(0))
        batch_rejected_inputs.append(rj_input_ids.squeeze(0))
        if ch_attn is not None:
            batch_chosen_masks.append(ch_attn.squeeze(0))
        else:
            batch_chosen_masks.append(None)
        if rj_attn is not None:
            batch_rejected_masks.append(rj_attn.squeeze(0))
        else:
            batch_rejected_masks.append(None)
        
        # Process batch when full or at the last step
        is_last_step = (steps is not None and step == steps - 1) or (steps is None and step == len(dataset) - 1)
        if len(batch_chosen_inputs) >= batch_size or is_last_step:
            # Stack into batch
            ch_input_ids = torch.stack(batch_chosen_inputs)
            rj_input_ids = torch.stack(batch_rejected_inputs)
            
            if batch_chosen_masks[0] is not None:
                ch_attn = torch.stack(batch_chosen_masks)
            else:
                ch_attn = None
            
            if batch_rejected_masks[0] is not None:
                rj_attn = torch.stack(batch_rejected_masks)
            else:
                rj_attn = None
            
            batch = {
                "chosen_input_ids": ch_input_ids,
                "chosen_attention_mask": ch_attn,
                "rejected_input_ids": rj_input_ids,
                "rejected_attention_mask": rj_attn,
            }
            
            # Forward pass
            chosen_rewards, rejected_rewards = concatenated_forward_rewards(
                reward_model,
                batch,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            )
            
            # Compute loss
            loss = reward_ranking_loss(chosen_rewards, rejected_rewards)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            loss_history.append({
                "step": step,
                "loss": loss_val,
                "chosen_reward_mean": chosen_rewards.mean().item(),
                "rejected_reward_mean": rejected_rewards.mean().item(),
            })
            
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                ch_rwd=f"{chosen_rewards.mean().item():.3f}",
                rj_rwd=f"{rejected_rewards.mean().item():.3f}",
            )
            
            # Clear batch
            batch_chosen_inputs = []
            batch_rejected_inputs = []
            batch_chosen_masks = []
            batch_rejected_masks = []
    
    # Process any remaining batch (shouldn't happen with current logic, but safety check)
    if len(batch_chosen_inputs) > 0:
        ch_input_ids = torch.stack(batch_chosen_inputs)
        rj_input_ids = torch.stack(batch_rejected_inputs)
        
        if batch_chosen_masks[0] is not None:
            ch_attn = torch.stack(batch_chosen_masks)
        else:
            ch_attn = None
        
        if batch_rejected_masks[0] is not None:
            rj_attn = torch.stack(batch_rejected_masks)
        else:
            rj_attn = None
        
        batch = {
            "chosen_input_ids": ch_input_ids,
            "chosen_attention_mask": ch_attn,
            "rejected_input_ids": rj_input_ids,
            "rejected_attention_mask": rj_attn,
        }
        
        chosen_rewards, rejected_rewards = concatenated_forward_rewards(
            reward_model,
            batch,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        )
        
        loss = reward_ranking_loss(chosen_rewards, rejected_rewards)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append({
            "step": step,
            "loss": loss_val,
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
        })
    
    # 5. Save reward model
    os.makedirs(output_dir, exist_ok=True)
    
    # Save only the reward head (base model is saved separately)
    # This avoids issues with LoRA structure differences
    torch.save(
        {
            "reward_head_state_dict": reward_model.reward_head.state_dict(),
            "base_model_name": model_name,
            "sft_lora_path": sft_lora_path,
        },
        os.path.join(output_dir, "reward_model.pt"),
    )
    
    # Also save base model and tokenizer separately for easier loading
    reward_model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save loss history
    loss_file = os.path.join(output_dir, "loss_history.json")
    with open(loss_file, "w") as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"✅ Reward model saved to: {output_dir}")
    print(f"✅ Loss history saved to: {loss_file}")


def load_reward_model(
    model_name: str,
    reward_model_path: str,
    sft_lora_path: str | None = None,
):
    """
    Load a trained reward model.
    
    Args:
        model_name: base model name
        reward_model_path: path to saved reward model checkpoint
        sft_lora_path: optional SFT LoRA path (if reward model was trained with LoRA)
    
    Returns:
        reward_model: RewardModel instance
        tokenizer: tokenizer
    """
    from src_utils import load_model
    from src_lora import prepare_lora
    
    # Load base model
    model, tokenizer = load_model(model_name)
    
    # Optionally load SFT LoRA
    if sft_lora_path is not None:
        model = prepare_lora(model, checkpoint_path=sft_lora_path)
    
    # Create reward model wrapper
    reward_model = RewardModel(model)
    
    # Load reward model state
    checkpoint = torch.load(
        os.path.join(reward_model_path, "reward_model.pt"),
        map_location=model.device,
    )
    
    # Load reward head (backward compatibility: check both old and new format)
    if "reward_head_state_dict" in checkpoint:
        # New format: only reward head is saved
        reward_model.reward_head.load_state_dict(checkpoint["reward_head_state_dict"])
    elif "reward_model_state_dict" in checkpoint:
        # Old format: full state dict (try to extract reward_head)
        saved_state = checkpoint["reward_model_state_dict"]
        reward_head_state = {}
        for key, value in saved_state.items():
            if key.startswith("reward_head."):
                reward_head_state[key.replace("reward_head.", "")] = value
        if reward_head_state:
            reward_model.reward_head.load_state_dict(reward_head_state, strict=False)
        else:
            print("Warning: Could not find reward_head in saved state_dict. Using random initialization.")
    else:
        print("Warning: Could not find reward_head_state_dict or reward_model_state_dict in checkpoint. Using random initialization.")
    
    reward_model.eval()
    return reward_model, tokenizer


@torch.no_grad()
def compute_reward(
    reward_model: RewardModel,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 512,
):
    """
    Compute reward for a prompt-response pair.
    
    Args:
        reward_model: RewardModel instance
        tokenizer: tokenizer
        prompt: input prompt
        response: model response
        max_length: max token length
    
    Returns:
        reward: scalar reward value
    """
    input_ids, attention_mask = build_reward_inputs(
        tokenizer,
        prompt,
        response,
        max_length,
        reward_model.base_model.device,
    )
    
    # Debug: check input hash to verify different inputs
    # Use only non-padding tokens for hash to compare actual content
    if tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    else:
        pad_token_id = 0
    
    # Use attention_mask if available (more reliable)
    if attention_mask is not None:
        seq_length = attention_mask.sum(dim=1).item()
        actual_length = seq_length
        non_padding_tokens = input_ids[0][:seq_length].cpu().numpy()
    else:
        # Fallback: check pad_token_id
        non_padding_mask = input_ids[0] != pad_token_id
        actual_length = non_padding_mask.sum().item()
        non_padding_tokens = input_ids[0][non_padding_mask].cpu().numpy()
        seq_length = input_ids.shape[1]
    
    input_hash = hash(non_padding_tokens.tobytes()) if len(non_padding_tokens) > 0 else 0
    
    # Also hash the response part specifically (last 100 tokens of non-padding)
    response_hash = hash(non_padding_tokens[-100:].tobytes()) if len(non_padding_tokens) > 100 else hash(non_padding_tokens.tobytes())
    
    # Debug: check what token position will be used for reward
    last_token_pos = actual_length - 1  # 0-indexed position that will be used
    print(f"[Reward Compute Debug] Response length: {len(response)}, Actual seq length: {actual_length}, Last token pos for reward: {last_token_pos}")
    print(f"[Reward Compute Debug] Full input hash: {input_hash}, Response part hash (last 100 tokens): {response_hash}")
    
    reward = reward_model(input_ids=input_ids, attention_mask=attention_mask)
    reward_value = reward.item()
    
    # Debug: print reward
    print(f"[Reward Compute Debug] Final reward: {reward_value:.6f}")
    
    return reward_value
