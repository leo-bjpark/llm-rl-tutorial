import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src_utils import load_shp_dataset, load_model, format_chat
from src_lora import prepare_lora

IGNORE_INDEX = -100


# ---------- helper functions ----------

def build_inputs_and_labels(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    device: torch.device,
):
    """
    TRL-style tokenization:
    - full chat = prompt + response
    - labels = input_ids clone
    - prompt tokens are masked with IGNORE_INDEX
    """

    # full conversation
    full_text = format_chat(
        tokenizer,
        prompt=prompt,
        response=response,
    )
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # prompt-only (with generation prompt to align boundary)
    prompt_only_text = format_chat(
        tokenizer,
        prompt=prompt,
        response=None,
        add_generation_prompt=True,
    )
    prompt_only = tokenizer(
        prompt_only_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = full["input_ids"].to(device)
    attention_mask = full.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = min(
        prompt_only["input_ids"].shape[1],
        input_ids.shape[1],
    )

    labels = input_ids.clone()
    labels[:, :prompt_len] = IGNORE_INDEX

    return input_ids, attention_mask, labels


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
):
    """
    Compute sequence log-probabilities ignoring labels == IGNORE_INDEX.
    Returns shape: (batch,)
    """
    # shift for causal LM
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    loss_mask = labels != IGNORE_INDEX
    labels_safe = labels.clone()
    labels_safe[~loss_mask] = 0

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(
        -1, labels_safe.unsqueeze(-1)
    ).squeeze(-1)

    token_logp = token_logp * loss_mask

    if average_log_prob:
        denom = loss_mask.sum(dim=-1).clamp_min(1)
        return token_logp.sum(dim=-1) / denom
    else:
        return token_logp.sum(dim=-1)


def concatenated_forward_logps(
    model,
    batch,
    average_log_prob: bool = False,
    pad_token_id: int = 0,
):
    """
    Concatenate chosen/rejected along batch dimension
    and run a single forward pass.
    Pads sequences to the same length before concatenation.
    """
    ch_input_ids = batch["chosen_input_ids"]
    rj_input_ids = batch["rejected_input_ids"]
    ch_labels = batch["chosen_labels"]
    rj_labels = batch["rejected_labels"]
    
    # Get max sequence length
    max_len = max(ch_input_ids.shape[1], rj_input_ids.shape[1])
    
    # Pad chosen sequences
    if ch_input_ids.shape[1] < max_len:
        pad_len = max_len - ch_input_ids.shape[1]
        ch_input_ids = F.pad(ch_input_ids, (0, pad_len), value=pad_token_id)
        ch_labels = F.pad(ch_labels, (0, pad_len), value=IGNORE_INDEX)
    
    # Pad rejected sequences
    if rj_input_ids.shape[1] < max_len:
        pad_len = max_len - rj_input_ids.shape[1]
        rj_input_ids = F.pad(rj_input_ids, (0, pad_len), value=pad_token_id)
        rj_labels = F.pad(rj_labels, (0, pad_len), value=IGNORE_INDEX)
    
    input_ids = torch.cat([ch_input_ids, rj_input_ids], dim=0)
    labels = torch.cat([ch_labels, rj_labels], dim=0)

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

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logps = get_batch_logps(
        outputs.logits,
        labels,
        average_log_prob=average_log_prob,
    )

    bsz = batch["chosen_input_ids"].shape[0]
    return logps[:bsz], logps[bsz:]


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float,
):
    """
    Standard DPO objective:
    -log σ( β * [ (πc - πr) - (refc - refr) ] )
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return (-F.logsigmoid(logits)).mean()


# ---------- main training function ----------

def run_dpo_lora(
    model_name: str,
    sft_lora_path: str,
    output_dir: str,
    split: str = "train",
    max_samples: int = 1000,
    lr: float = 1e-4,
    beta: float = 0.1,
    max_length: int = 512,
    steps: int = 300,
    average_log_prob: bool = True,
):
    """
    Stable TRL-style DPO training with LoRA.
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
    model.config.use_cache = False  # important for training stability

    # 3. Load frozen reference model (same tokenizer assumption)
    ref_model, _ = load_model(model_name)
    ref_model.eval()
    ref_model.config.use_cache = False
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    loss_history = []
    pbar = tqdm(dataset, desc="DPO training")

    for step, sample in enumerate(pbar):
        if step >= steps:
            break

        # --- build chosen / rejected inputs ---
        ch_input_ids, ch_attn, ch_labels = build_inputs_and_labels(
            tokenizer,
            sample["prompt"],
            sample["chosen"],
            max_length,
            model.device,
        )

        rj_input_ids, rj_attn, rj_labels = build_inputs_and_labels(
            tokenizer,
            sample["prompt"],
            sample["rejected"],
            max_length,
            model.device,
        )

        batch = {
            "chosen_input_ids": ch_input_ids,
            "chosen_attention_mask": ch_attn,
            "chosen_labels": ch_labels,
            "rejected_input_ids": rj_input_ids,
            "rejected_attention_mask": rj_attn,
            "rejected_labels": rj_labels,
        }

        # --- reference log-probs ---
        with torch.no_grad():
            ref_ch_logps, ref_rj_logps = concatenated_forward_logps(
                ref_model,
                batch,
                average_log_prob=average_log_prob,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            )

        # --- policy log-probs ---
        pol_ch_logps, pol_rj_logps = concatenated_forward_logps(
            model,
            batch,
            average_log_prob=average_log_prob,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        )

        # --- DPO loss ---
        loss = dpo_loss(
            pol_ch_logps,
            pol_rj_logps,
            ref_ch_logps,
            ref_rj_logps,
            beta=beta,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append({
            "step": step,
            "loss": loss_val,
        })

        pbar.set_postfix(loss=f"{loss_val:.4f}")

    # 4. Save LoRA checkpoint
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)

    print(f"✅ DPO LoRA model saved to: {output_dir}")
