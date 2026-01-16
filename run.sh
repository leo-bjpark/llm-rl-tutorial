

model_name="google/gemma-3-4b-it"

split="train"
max_samples=1000
lr=1e-4
max_length=512
steps=300

# SFT (Supervised Fine-Tuning)
algorithm="sft"
batch_size=2
epochs=1
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
sft_lora_path="checkpoints/google/gemma-3-4b-it-sft-lora"

# DPO (Direct Preference Optimization)
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
algorithm="ppo"
reward_model_path="checkpoints/google/gemma-3-4b-it-rm-lora"
kl_coef=0.1
clip_epsilon=0.2
ppo_epochs=4
max_new_tokens=128
temperature=0.7
top_p=0.9
python train.py \
    --model_name $model_name \
    --algorithm $algorithm \
    --reward_model_path $reward_model_path \
    --sft_lora_path $sft_lora_path \
    --split $split \
    --max_samples $max_samples \
    --batch_size $batch_size \
    --lr $lr \
    --kl_coef $kl_coef \
    --clip_epsilon $clip_epsilon \
    --max_new_tokens $max_new_tokens \
    --max_length $max_length \
    --steps $steps \
    --ppo_epochs $ppo_epochs \
    --temperature $temperature \
    --top_p $top_p

# GRPO (Group Relative Policy Optimization)
algorithm="grpo"
num_responses=4
use_dataset_responses=false
python train.py \
    --model_name $model_name \
    --algorithm $algorithm \
    --sft_lora_path $sft_lora_path \
    --split $split \
    --max_samples $max_samples \
    --lr $lr \
    --beta $beta \
    --max_length $max_length \
    --steps $steps \
    --num_responses $num_responses \
    --max_new_tokens $max_new_tokens \
    --temperature $temperature \
    --top_p $top_p \
    --average_log_prob $average_log_prob \
    --use_dataset_responses $use_dataset_responses \
    --reward_model_path $reward_model_path