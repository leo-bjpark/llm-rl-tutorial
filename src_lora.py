
# ============================================================
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from src_utils import load_model, load_shp_dataset, format_chat
import json
import os


def run_sft_lora(
    model_name: str,
    output_dir: str,
    split: str = "train",
    max_samples: int = 2000,
    batch_size: int = 2,
    lr: float = 2e-4,
    epochs: int = 1,
    max_length: int = 512,
    lora_checkpoint: str | None = None,
):
    """
    Run supervised fine-tuning (SFT) with LoRA using SHP dataset.
    Only the chosen response is used.

    Args:
        model_name: base model name (e.g., google/gemma-3-2b-it)
        output_dir: directory to save LoRA checkpoint
        split: dataset split
        max_samples: number of training samples
        batch_size: training batch size
        lr: learning rate
        epochs: number of epochs
        max_length: max token length
        lora_checkpoint: optional LoRA checkpoint to resume from
    """

    # 1. Load dataset
    dataset = load_shp_dataset(
        split=split,
        max_samples=max_samples,
    )

    # 2. Load model + tokenizer
    model, tokenizer = load_model(model_name)
    model = prepare_lora(model, checkpoint_path=lora_checkpoint)
    model.train()

    # 3. Tokenization using chat template
    def tokenize_fn(example):
        text = format_chat(
            tokenizer,
            prompt=example["prompt"],
            response=example["chosen"],
        )

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
    )

    # 4. DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # 6. Training loop
    loss_history = []
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = {
                k: v.to(model.device)
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            loss_history.append({
                "epoch": epoch,
                "step": step,
                "loss": loss_value
            })
            pbar.set_postfix(loss=f"{loss_value:.4f}")

    # 7. Save LoRA checkpoint
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save loss history
    loss_file = os.path.join(output_dir, "loss_history.json")
    with open(loss_file, "w") as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"✅ SFT LoRA model saved to: {output_dir}")
    print(f"✅ Loss history saved to: {loss_file}")




def prepare_lora(model, checkpoint_path=None):
    """
    If checkpoint_path is provided:
        - Load LoRA adapter from checkpoint
    Else:
        - Initialize a new LoRA configuration
    """

    if checkpoint_path is not None:
        # Load existing LoRA adapter
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=True,   # important for further tuning (DPO / GRPO)
        )
    else:
        # Create new LoRA adapter
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model

def load_lora_model(model_name, lora_path):
    model, tokenizer = load_model(model_name)
    model = prepare_lora(model, checkpoint_path=lora_path)
    model.eval()
    return model, tokenizer