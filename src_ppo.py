import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src_utils import load_shp_dataset, load_model, format_chat, generate_response
from src_lora import prepare_lora
from src_rm import RewardModel, load_reward_model, build_reward_inputs, compute_reward
from src_dpo import get_batch_logps, build_inputs_and_labels

IGNORE_INDEX = -100


def compute_kl_penalty(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    kl_coef: float,
):
    """
    Compute KL divergence penalty between policy and reference model.
    
    Args:
        policy_logps: (batch_size,) log probabilities from policy model
        ref_logps: (batch_size,) log probabilities from reference model
        kl_coef: KL penalty coefficient
    
    Returns:
        kl_penalty: scalar tensor
    """
    kl = policy_logps - ref_logps
    return kl_coef * kl.mean()


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor | None = None,
    gamma: float = 1.0,
    lam: float = 0.95,
):
    """
    Compute advantages using GAE (Generalized Advantage Estimation) if values provided,
    otherwise use rewards directly.
    
    Args:
        rewards: (batch_size,) reward values
        values: (batch_size,) value estimates (optional)
        gamma: discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: (batch_size,)
    """
    if values is None:
        # Simple case: use rewards as advantages
        return rewards
    
    # GAE computation (simplified for single-step case)
    # For multi-step, would need to compute returns and advantages properly
    # Here we use a simple advantage = reward - value
    advantages = rewards - values
    return advantages


def ppo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
):
    """
    Compute clipped PPO objective.
    
    Args:
        policy_logps: (batch_size,) log probabilities from current policy
        old_logps: (batch_size,) log probabilities from old policy (when action was taken)
        advantages: (batch_size,) advantage estimates
        clip_epsilon: clipping parameter
    
    Returns:
        loss: scalar tensor
    """
    # Compute importance sampling ratio
    ratio = torch.exp(policy_logps - old_logps)
    
    # Clipped objective
    loss1 = ratio * advantages
    loss2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    # Take minimum (pessimistic bound)
    loss = -torch.min(loss1, loss2).mean()
    
    return loss


def generate_response_with_logprobs(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate a response and return the response string along with token information.
    
    Returns:
        response: generated response string
        prompt_input_ids: tokenized prompt
        full_input_ids: full sequence (prompt + generated)
    """
    # Get device from model (handle device_map="auto" case)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # Fallback if model has no parameters (shouldn't happen)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Format prompt with generation template
    prompt_text = format_chat(
        tokenizer,
        prompt=prompt,
        response=None,
        add_generation_prompt=True,
    )
    
    # Tokenize prompt
    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    
    prompt_input_ids = prompt_tokens["input_ids"]
    prompt_len = prompt_input_ids.shape[1]
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    
    # Extract generated tokens (excluding prompt)
    generated_ids = outputs[0, prompt_len:]
    
    # Decode response
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response, prompt_input_ids.squeeze(0), outputs[0]


def compute_reward_single(
    reward_model: RewardModel,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 512,
):
    """
    Compute reward for a single prompt-response pair.
    
    Returns:
        reward: scalar reward value
    """
    return compute_reward(reward_model, tokenizer, prompt, response, max_length)


def run_ppo_training(
    model_name: str,
    reward_model_path: str,
    sft_lora_path: str,
    output_dir: str,
    split: str = "train",
    max_samples: int = 100,
    batch_size: int = 4,
    lr: float = 1e-5,
    kl_coef: float = 0.1,
    clip_epsilon: float = 0.2,
    max_new_tokens: int = 128,
    max_length: int = 512,
    steps: int | None = None,
    ppo_epochs: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Train a policy model using PPO with a reward model.
    
    Args:
        model_name: base model name
        reward_model_path: path to trained reward model checkpoint
        sft_lora_path: path to SFT LoRA checkpoint (initial policy)
        output_dir: directory to save PPO checkpoint
        split: dataset split
        max_samples: number of training samples
        batch_size: batch size for generation and training
        lr: learning rate
        kl_coef: KL penalty coefficient
        clip_epsilon: PPO clipping parameter
        max_new_tokens: maximum tokens to generate
        max_length: maximum sequence length
        steps: number of training steps (if None, uses all samples)
        ppo_epochs: number of PPO update epochs per batch
        temperature: sampling temperature
        top_p: top-p sampling parameter
    """
    
    # 1. Load dataset
    dataset = load_shp_dataset(
        split=split,
        max_samples=max_samples,
    )
    
    # 2. Load policy model (with SFT LoRA)
    policy_model, tokenizer = load_model(model_name)
    policy_model = prepare_lora(policy_model, checkpoint_path=sft_lora_path)
    policy_model.train()
    policy_model.config.use_cache = False
    
    # 3. Load frozen reference model (same as initial policy)
    ref_model, _ = load_model(model_name)
    ref_model = prepare_lora(ref_model, checkpoint_path=sft_lora_path)
    ref_model.eval()
    ref_model.config.use_cache = False
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # 4. Load frozen reward model
    reward_model, _ = load_reward_model(
        model_name,
        reward_model_path,
        sft_lora_path=None,  # Reward model should be trained on base model
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=lr,
    )
    
    # 6. Training loop
    loss_history = []
    pbar = tqdm(dataset, desc="PPO Training")
    
    for step, sample in enumerate(pbar):
        if steps is not None and step >= steps:
            break
        
        prompt = sample["prompt"]
        
        # Generate response
        response, prompt_input_ids, full_input_ids = generate_response_with_logprobs(
            policy_model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Compute reward
        reward = compute_reward_single(
            reward_model,
            tokenizer,
            prompt,
            response,
            max_length=max_length,
        )
        
        # Get device from policy model for input preparation
        try:
            policy_device = next(policy_model.parameters()).device
        except StopIteration:
            policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build inputs for log probability computation
        # We need to compute log probs for the generated response
        ch_input_ids, ch_attn, ch_labels = build_inputs_and_labels(
            tokenizer,
            prompt,
            response,
            max_length,
            policy_device,
        )
        
        # Compute old log probabilities (from current policy, before update)
        with torch.no_grad():
            old_outputs = policy_model(
                input_ids=ch_input_ids,
                attention_mask=ch_attn,
            )
            old_logps = get_batch_logps(
                old_outputs.logits,
                ch_labels,
                average_log_prob=False,
            )
        
        # Compute reference log probabilities
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=ch_input_ids,
                attention_mask=ch_attn,
            )
            ref_logps = get_batch_logps(
                ref_outputs.logits,
                ch_labels,
                average_log_prob=False,
            )
        
        # PPO update epochs
        for ppo_epoch in range(ppo_epochs):
            # Compute current policy log probabilities
            policy_outputs = policy_model(
                input_ids=ch_input_ids,
                attention_mask=ch_attn,
            )
            policy_logps = get_batch_logps(
                policy_outputs.logits,
                ch_labels,
                average_log_prob=False,
            )
            
            # Ensure all tensors are on the same device and dtype
            # Get device and dtype from policy_logps
            policy_logps_device = policy_logps.device
            policy_logps_dtype = policy_logps.dtype
            
            # Move tensors to match policy_logps
            old_logps = old_logps.to(device=policy_logps_device, dtype=policy_logps_dtype)
            ref_logps = ref_logps.to(device=policy_logps_device, dtype=policy_logps_dtype)
            
            # Create reward tensor with matching device and dtype
            reward_tensor = torch.tensor(
                reward, 
                device=policy_logps_device, 
                dtype=policy_logps_dtype
            )
            
            # Compute advantages (using reward directly)
            advantages = compute_advantages(reward_tensor.unsqueeze(0))
            
            # Compute PPO loss
            ppo_loss_val = ppo_loss(
                policy_logps,
                old_logps,
                advantages,
                clip_epsilon=clip_epsilon,
            )
            
            # Compute KL penalty
            kl_penalty = compute_kl_penalty(
                policy_logps,
                ref_logps,
                kl_coef=kl_coef,
            )
            
            # Total loss
            total_loss = ppo_loss_val + kl_penalty
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
        
        # Logging
        loss_history.append({
            "step": step,
            "reward": reward,
            "ppo_loss": ppo_loss_val.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": total_loss.item(),
        })
        
        pbar.set_postfix(
            reward=f"{reward:.3f}",
            loss=f"{total_loss.item():.4f}",
        )
    
    # 7. Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"✅ PPO model saved to: {output_dir}")
