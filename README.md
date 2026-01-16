# LLM RL Tutorial 


This repository aims to build a conceptually clean and minimal understanding of preference-based tuning for large language models, focusing on how reward signals, generation, and optimization interact.

The project is organized into three conceptual parts.


# Learning Comparison

## Learning Comparison

| Algorithm | What is Optimized | Supervision Signal | Reward Model Needed | Key Idea | Computational Cost |
|----------|------------------|--------------------|---------------------|----------|--------------------|
| **SFT** | Likelihood of human-written responses | Chosen responses only | No | Learn by imitation (what humans say) | Low |
| **DPO** | Relative preference between responses | Chosen vs rejected pairs | No | Directly align policy with preferences using a reference model | Medium |
| **RM** | Scalar reward function | Chosen vs rejected pairs | Yes (this model) | Learn a reward signal that represents human preference | Low |
| **PPO** | Expected reward under policy | Scalar rewards | Yes | Reinforcement learning with exploration and KL regularization | High |
| **GRPO** | Relative ordering within a response group | Ranked / compared generations | Optional (requires only relative preference signals, not scalar rewards) | Use group-wise comparisons for preference optimization | Medium–High |


## Conceptual Differences

| Aspect | SFT | DPO | RM | PPO | GRPO |
|------|-----|-----|----|-----|------|
| Uses preferences | ❌ | ✅ | ✅ | ✅ | ✅ |
| Explicit RL loop | ❌ | ❌ | ❌ | ✅ | ❌ |
| Requires reference policy | ❌ | ✅ | ❌ | ✅ | ✅ |
| Online generation during training | ❌ | ❌ | ❌ | ✅ | ✅ |
| Optimization stability | High | High | High | Medium | High |


```
bash run.sh 

python app.py
```