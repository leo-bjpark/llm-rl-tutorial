import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src_utils import load_shp_dataset, load_model, format_chat
from src_lora import prepare_lora
from src_dpo import get_batch_logps, build_inputs_and_labels, concatenated_forward_logps
from src_rm import RewardModel, load_reward_model, compute_reward, build_reward_inputs

IGNORE_INDEX = -100


def grpo_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    beta: float,
):
    """
    GRPO (Group Relative Policy Optimization) loss.
    
    For a group of responses, computes ranking loss based on rewards.
    Similar to DPO but works with multiple responses in a group.
    
    Loss = -log σ( β * [ (πi - πj) - (refi - refj) ] ) for all pairs where reward_i > reward_j
    
    Args:
        policy_logps: (group_size,) log probabilities from policy model
        ref_logps: (group_size,) log probabilities from reference model
        rewards: (group_size,) reward values for each response
        beta: temperature parameter
    
    Returns:
        loss: scalar tensor
    """
    group_size = policy_logps.shape[0]
    
    if group_size < 2:
        # Need at least 2 responses for ranking
        return torch.tensor(0.0, device=policy_logps.device, dtype=policy_logps.dtype, requires_grad=True)
    
    # Compute all pairwise comparisons
    # For each pair (i, j) where reward[i] > reward[j]
    losses = []
    
    for i in range(group_size):
        for j in range(i + 1, group_size):
            if rewards[i] > rewards[j]:
                # Response i is better than j
                pi_logratio = policy_logps[i] - policy_logps[j]
                ref_logratio = ref_logps[i] - ref_logps[j]
                logits = beta * (pi_logratio - ref_logratio)
                losses.append(-F.logsigmoid(logits))
            elif rewards[j] > rewards[i]:
                # Response j is better than i
                pi_logratio = policy_logps[j] - policy_logps[i]
                ref_logratio = ref_logps[j] - ref_logps[i]
                logits = beta * (pi_logratio - ref_logratio)
                losses.append(-F.logsigmoid(logits))
    
    if len(losses) == 0:
        # All rewards are equal, return zero loss
        return torch.tensor(0.0, device=policy_logps.device, dtype=policy_logps.dtype, requires_grad=True)
    
    return torch.stack(losses).mean()


def generate_responses_group(
    model,
    tokenizer,
    prompt: str,
    num_responses: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate multiple responses for a single prompt.
    
    Returns:
        responses: list of generated response strings
    """
    # Get device from model
    try:
        device = next(model.parameters()).device
    except StopIteration:
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
    
    responses = []
    for _ in range(num_responses):
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
        
        prompt_len = prompt_tokens["input_ids"].shape[1]
        generated_ids = outputs[0, prompt_len:]
        
        # Decode response
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        responses.append(response)
    
    return responses


def compute_group_logps(
    model,
    tokenizer,
    prompt: str,
    responses: list[str],
    max_length: int,
    average_log_prob: bool = False,
):
    """
    Compute log probabilities for a group of responses.
    Optimized to process all responses in a single batch.
    
    Returns:
        logps: (group_size,) tensor of log probabilities
    """
    # Get device from model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(responses) == 0:
        return torch.tensor([], device=device)
    
    # Build inputs for all responses
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for response in responses:
        input_ids, attention_mask, labels = build_inputs_and_labels(
            tokenizer,
            prompt,
            response,
            max_length,
            device,
        )
        input_ids_list.append(input_ids.squeeze(0))
        labels_list.append(labels.squeeze(0))
        if attention_mask is not None:
            attention_mask_list.append(attention_mask.squeeze(0))
        else:
            attention_mask_list.append(None)
    
    # Find max length for padding
    max_len = max(ids.shape[0] for ids in input_ids_list)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Pad and stack all sequences
    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []
    
    for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
        if input_ids.shape[0] < max_len:
            pad_len = max_len - input_ids.shape[0]
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            labels = F.pad(labels, (0, pad_len), value=IGNORE_INDEX)
        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        
        attn = attention_mask_list[i]
        if attn is not None:
            if attn.shape[0] < max_len:
                pad_len = max_len - attn.shape[0]
                attn = F.pad(attn, (0, pad_len), value=0)
            padded_attention_mask.append(attn)
    
    # Stack into batch
    batch_input_ids = torch.stack(padded_input_ids)
    batch_labels = torch.stack(padded_labels)
    
    if padded_attention_mask and padded_attention_mask[0] is not None:
        batch_attention_mask = torch.stack(padded_attention_mask)
    else:
        batch_attention_mask = None
    
    # Single forward pass for all responses
    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
    )
    
    # Compute log probabilities for all responses at once
    logps = get_batch_logps(
        outputs.logits,
        batch_labels,
        average_log_prob=average_log_prob,
    )
    
    return logps


def run_grpo_lora(
    model_name: str,
    sft_lora_path: str,
    output_dir: str,
    split: str = "train",
    max_samples: int = 1000,
    lr: float = 1e-4,
    beta: float = 0.1,
    max_length: int = 512,
    steps: int = 300,
    num_responses: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    average_log_prob: bool = True,
    use_dataset_responses: bool = False,
    reward_model_path: str | None = None,
):
    """
    Train a policy model using GRPO (Group Relative Policy Optimization).
    
    GRPO generates multiple responses per prompt and learns to rank them
    based on rewards or preferences. Unlike DPO which compares chosen/rejected
    pairs, GRPO works with groups of responses.
    
    Args:
        model_name: base model name
        sft_lora_path: path to SFT LoRA checkpoint (initial policy)
        output_dir: directory to save GRPO checkpoint
        split: dataset split
        max_samples: number of training samples
        lr: learning rate
        beta: temperature parameter for GRPO loss
        max_length: max token length
        steps: number of training steps
        num_responses: number of responses to generate per prompt
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_p: top-p sampling parameter
        average_log_prob: whether to average log probabilities by sequence length
        use_dataset_responses: if True, use chosen/rejected from dataset as responses
                              if False, generate responses online
        reward_model_path: optional path to reward model for computing rewards
                          if None, uses simple heuristic (length-based)
    """
    
    # 1. Load dataset
    dataset = load_shp_dataset(
        split=split,
        max_samples=max_samples,
    )
    
    # 2. Load policy model + tokenizer + SFT LoRA
    model, tokenizer = load_model(model_name)
    model = prepare_lora(model, checkpoint_path=sft_lora_path)
    model.train()
    model.config.use_cache = False
    
    # 3. Load frozen reference model
    ref_model, _ = load_model(model_name)
    ref_model = prepare_lora(ref_model, checkpoint_path=sft_lora_path)
    ref_model.eval()
    ref_model.config.use_cache = False
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # 3.5. Load reward model if provided
    reward_model = None
    if reward_model_path is not None:
        reward_model, _ = load_reward_model(
            model_name,
            reward_model_path,
            sft_lora_path=None,
        )
        reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad = False
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )
    
    # 5. Training loop
    loss_history = []
    pbar = tqdm(dataset, desc="GRPO training")
    
    for step, sample in enumerate(pbar):
        if step >= steps:
            break
        
        prompt = sample["prompt"]
        
        if use_dataset_responses:
            # Use dataset responses (chosen + rejected + generate more if needed)
            responses = [sample["chosen"], sample["rejected"]]
            # Generate additional responses if needed
            if num_responses > 2:
                additional = generate_responses_group(
                    model,
                    tokenizer,
                    prompt,
                    num_responses=num_responses - 2,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                responses.extend(additional)
            
            # Assign rewards: chosen=1.0, rejected=0.0, generated=0.5 (neutral)
            rewards = torch.tensor(
                [1.0, 0.0] + [0.5] * (num_responses - 2),
                device=next(model.parameters()).device,
                dtype=torch.float32,
            )
        else:
            # Generate all responses online
            responses = generate_responses_group(
                model,
                tokenizer,
                prompt,
                num_responses=num_responses,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Compute rewards
            if reward_model is not None:
                # Use reward model to compute rewards (batch processing)
                try:
                    device = next(model.parameters()).device
                except StopIteration:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Build inputs for all responses
                input_ids_list = []
                attention_mask_list = []
                
                for response in responses:
                    input_ids, attention_mask = build_reward_inputs(
                        tokenizer,
                        prompt,
                        response,
                        max_length,
                        device,
                    )
                    input_ids_list.append(input_ids.squeeze(0))
                    if attention_mask is not None:
                        attention_mask_list.append(attention_mask.squeeze(0))
                    else:
                        attention_mask_list.append(None)
                
                # Pad and stack
                max_len = max(ids.shape[0] for ids in input_ids_list)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                
                padded_input_ids = []
                padded_attention_mask = []
                
                for i, input_ids in enumerate(input_ids_list):
                    if input_ids.shape[0] < max_len:
                        pad_len = max_len - input_ids.shape[0]
                        input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
                    padded_input_ids.append(input_ids)
                    
                    attn = attention_mask_list[i]
                    if attn is not None:
                        if attn.shape[0] < max_len:
                            pad_len = max_len - attn.shape[0]
                            attn = F.pad(attn, (0, pad_len), value=0)
                        padded_attention_mask.append(attn)
                
                batch_input_ids = torch.stack(padded_input_ids)
                if padded_attention_mask and padded_attention_mask[0] is not None:
                    batch_attention_mask = torch.stack(padded_attention_mask)
                else:
                    batch_attention_mask = None
                
                # Single forward pass for all rewards
                with torch.no_grad():
                    rewards = reward_model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
            else:
                # Use simple heuristic: length-normalized rewards
                # (In practice, use a reward model for better results)
                response_lengths = [len(r.split()) for r in responses]
                max_len = max(response_lengths) if response_lengths else 1
                rewards = torch.tensor(
                    [len(r.split()) / max_len for r in responses],
                    device=next(model.parameters()).device,
                    dtype=torch.float32,
                )
        
        # Compute policy log probabilities
        policy_logps = compute_group_logps(
            model,
            tokenizer,
            prompt,
            responses,
            max_length,
            average_log_prob=average_log_prob,
        )
        
        # Compute reference log probabilities
        with torch.no_grad():
            ref_logps = compute_group_logps(
                ref_model,
                tokenizer,
                prompt,
                responses,
                max_length,
                average_log_prob=average_log_prob,
            )
        
        # Ensure all tensors are on the same device and dtype
        policy_logps_device = policy_logps.device
        policy_logps_dtype = policy_logps.dtype
        
        ref_logps = ref_logps.to(device=policy_logps_device, dtype=policy_logps_dtype)
        
        # Get device and dtype for rewards
        try:
            model_param = next(model.parameters())
            rewards_device = model_param.device
            rewards_dtype = model_param.dtype
        except StopIteration:
            rewards_device = policy_logps_device
            rewards_dtype = torch.float32
        
        rewards = rewards.to(device=rewards_device, dtype=rewards_dtype)
        
        # Compute GRPO loss
        loss = grpo_loss(
            policy_logps,
            ref_logps,
            rewards,
            beta=beta,
        )
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append({
            "step": step,
            "loss": loss_val,
            "num_responses": len(responses),
        })
        
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    
    # 6. Save LoRA checkpoint
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"✅ GRPO LoRA model saved to: {output_dir}")
