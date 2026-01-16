import torch
from datasets import load_dataset

def load_shp_dataset(
    split: str = "train",
    max_samples: int | None = 1000,
    seed: int = 42,
):
    """
    Load Stanford Human Preferences (SHP) dataset
    and convert it into (prompt, chosen, rejected) format.

    Returns:
        List[Dict[str, str]]
    """

    dataset = load_dataset("stanfordnlp/SHP", split=split)

    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    def convert(example):
        if example["labels"] == 1:
            chosen = example["human_ref_A"]
            rejected = example["human_ref_B"]
        else:
            chosen = example["human_ref_B"]
            rejected = example["human_ref_A"]

        return {
            "prompt": example["history"],
            "chosen": chosen,
            "rejected": rejected,
        }

    dataset = dataset.map(
        convert,
        remove_columns=dataset.column_names,
    )

    return dataset

def format_chat(
    tokenizer,
    prompt: str,
    response: str | None = None,
    add_generation_prompt: bool = False,
):
    """
    Format prompt/response using the model's chat template.

    If response is None:
        - Used for generation (policy sampling)
    Else:
        - Used for training (SFT, DPO, RM)

    Returns:
        str (formatted text)
    """

    messages = [{"role": "user", "content": prompt}]

    if response is not None:
        messages.append(
            {"role": "assistant", "content": response}
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

# ==== Model Loading and LoRA Configuration ====
from transformers import AutoModelForCausalLM, AutoTokenizer
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate a response from the model given a prompt,
    using the model's chat template.
    """

    prompt_text = format_chat(
        tokenizer,
        prompt=prompt,
        response=None,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return generated.strip()
