import argparse
from src_lora import run_sft_lora
from src_dpo import run_dpo_lora
from src_ppo import run_ppo_training
from src_grpo import run_grpo_lora
from src_rm import run_reward_model_training

def main():
    parser = argparse.ArgumentParser(description="Train LLM with SFT, DPO, PPO, GRPO, or RM")
    parser.add_argument("--algorithm", type=str, default="sft", choices=["sft", "dpo", "ppo", "grpo", "rm"],
                       help="Training algorithm to use")
    
    # Common arguments
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it",
                       help="Base model name")
    parser.add_argument("--sft_lora_path", type=str, default=None,
                       help="Path to SFT LoRA checkpoint (auto-generated if None)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of training samples")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # SFT-specific
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size (SFT, PPO)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs (SFT)")
    
    # DPO/GRPO-specific
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Temperature parameter (DPO, GRPO)")
    parser.add_argument("--steps", type=int, default=300,
                       help="Number of training steps (DPO, PPO, GRPO)")
    parser.add_argument("--average_log_prob", type=lambda x: (str(x).lower() == 'true'),
                       default=True,
                       help="Average log probabilities by sequence length (DPO, GRPO)")
    
    # PPO-specific
    parser.add_argument("--kl_coef", type=float, default=0.1,
                       help="KL penalty coefficient (PPO)")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                       help="PPO clipping parameter")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                       help="Number of PPO update epochs per batch")
    
    # GRPO-specific
    parser.add_argument("--num_responses", type=int, default=4,
                       help="Number of responses per prompt (GRPO)")
    parser.add_argument("--use_dataset_responses", type=lambda x: (str(x).lower() == 'true'),
                       default=False,
                       help="Use dataset chosen/rejected responses (GRPO)")
    
    # Generation parameters (PPO, GRPO)
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    
    # Reward model (PPO, GRPO)
    parser.add_argument("--reward_model_path", type=str, default=None,
                       help="Path to reward model checkpoint (PPO, GRPO)")
    
    args = parser.parse_args()
    
    # Auto-generate paths based on model_name and algorithm
    # Convert model_name to safe directory name (replace / with -)
    model_dir = args.model_name.replace("/", "-")
    base_checkpoint_dir = f"checkpoints/{model_dir}"
    
    algorithm_suffix = {
        "sft": "sft-lora",
        "dpo": "dpo-lora",
        "ppo": "ppo-lora",
        "grpo": "grpo-lora",
        "rm": "rm-lora",
    }
    args.output_dir = f"{base_checkpoint_dir}/{algorithm_suffix[args.algorithm]}"
    
    # Auto-generate sft_lora_path if not provided (only for non-SFT algorithms)
    if args.algorithm != "sft" and args.sft_lora_path is None:
        args.sft_lora_path = f"{base_checkpoint_dir}/sft-lora"
    
    # Auto-generate reward_model_path for PPO/GRPO if not provided
    if args.algorithm in ["ppo", "grpo"] and args.reward_model_path is None:
        args.reward_model_path = f"{base_checkpoint_dir}/rm-lora"

    if args.algorithm == "sft":
        run_sft_lora(
            model_name=args.model_name,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            max_length=args.max_length,
            lora_checkpoint=None,  # SFT doesn't need a checkpoint to start from
        )
    elif args.algorithm == "dpo":
        run_dpo_lora(
            model_name=args.model_name,
            sft_lora_path=args.sft_lora_path,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            lr=args.lr,
            beta=args.beta,
            max_length=args.max_length,
            steps=args.steps,
            average_log_prob=args.average_log_prob,
        )
    elif args.algorithm == "ppo":
        # reward_model_path is already auto-generated above if None
        run_ppo_training(
            model_name=args.model_name,
            reward_model_path=args.reward_model_path,
            sft_lora_path=args.sft_lora_path,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            lr=args.lr,
            kl_coef=args.kl_coef,
            clip_epsilon=args.clip_epsilon,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            steps=args.steps,
            ppo_epochs=args.ppo_epochs,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.algorithm == "grpo":
        run_grpo_lora(
            model_name=args.model_name,
            sft_lora_path=args.sft_lora_path,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            lr=args.lr,
            beta=args.beta,
            max_length=args.max_length,
            steps=args.steps,
            num_responses=args.num_responses,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            average_log_prob=args.average_log_prob,
            use_dataset_responses=args.use_dataset_responses,
            reward_model_path=args.reward_model_path,
        )
    elif args.algorithm == "rm":
        run_reward_model_training(
            model_name=args.model_name,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            steps=args.steps,
            sft_lora_path=args.sft_lora_path,
        )

if __name__ == "__main__":
    main()
