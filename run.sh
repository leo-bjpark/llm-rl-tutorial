

model_name="google/gemma-3-4b-it"

split="train"
max_samples=1000
lr=1e-4
max_length=128
steps=3000

# SFT (Supervised Fine-Tuning)
algorithm="sft"
batch_size=2
epochs=5
python train.py \
    --model_name $model_name \
    --algorithm $algorithm \
    --split $split \
    --max_samples $max_samples \
    --lr $lr \
    --max_length $max_length \
    --batch_size $batch_size \
    --epochs $epochs

# After SFT training, set the path for other algorithms
# Note: The path will be auto-generated if not provided, but you can set it explicitly here
sft_lora_path="checkpoints/$model_name/sft-lora"

# DPO (Direct Preference Optimization)
# Requires: SFT checkpoint must exist at sft_lora_path
algorithm="dpo"
beta=0.1
average_log_prob=true
python train.py \
    --model_name $model_name \
    --algorithm $algorithm \
    --sft_lora_path $sft_lora_path \
    --split $split \
    --max_samples $max_samples \
    --lr $lr \
    --max_length $max_length \
    --steps $steps \
    --beta $beta \
    --average_log_prob $average_log_prob

# RM (Reward Model)
# Requires: SFT checkpoint must exist at sft_lora_path
algorithm="rm"
python train.py \
    --model_name $model_name \
    --algorithm $algorithm \
    --sft_lora_path $sft_lora_path \
    --split $split \
    --max_samples $max_samples \
    --lr $lr \
    --max_length $max_length \
    --steps $steps \
    --batch_size $batch_size

# PPO (Proximal Policy Optimization)
# Requires: SFT checkpoint and RM checkpoint must exist
# Fast test settings
algorithm="ppo"
reward_model_path="checkpoints/$model_name/rm-lora"
kl_coef=0.1
clip_epsilon=0.2
ppo_epochs=1
ppo_max_samples=10
ppo_steps=5
ppo_max_new_tokens=32
ppo_batch_size=1
temperature=0.7
top_p=0.9
python train.py \
    --model_name $model_name \
    --algorithm $algorithm \
    --reward_model_path $reward_model_path \
    --sft_lora_path $sft_lora_path \
    --split $split \
    --max_samples $ppo_max_samples \
    --batch_size $ppo_batch_size \
    --lr $lr \
    --kl_coef $kl_coef \
    --clip_epsilon $clip_epsilon \
    --max_new_tokens $ppo_max_new_tokens \
    --max_length $max_length \
    --steps $ppo_steps \
    --ppo_epochs $ppo_epochs \
    --temperature $temperature \
    --top_p $top_p

# GRPO (Group Relative Policy Optimization)
# Requires: SFT checkpoint (RM checkpoint is optional)
# Fast test settings
algorithm="grpo"
grpo_max_samples=10
grpo_steps=5
grpo_max_new_tokens=32
num_responses=2
use_dataset_responses=false
# reward_model_path="checkpoints/$model_name/rm-lora"  # Uncomment to use reward model
python train.py \
    --model_name $model_name \
    --algorithm $algorithm \
    --sft_lora_path $sft_lora_path \
    --split $split \
    --max_samples $grpo_max_samples \
    --lr $lr \
    --beta $beta \
    --max_length $max_length \
    --steps $grpo_steps \
    --num_responses $num_responses \
    --max_new_tokens $grpo_max_new_tokens \
    --temperature $temperature \
    --top_p $top_p \
    --average_log_prob $average_log_prob \
    --use_dataset_responses $use_dataset_responses
    # --reward_model_path $reward_model_path  # Uncomment if reward_model_path is set above