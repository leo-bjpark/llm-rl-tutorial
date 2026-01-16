# LLM RL Tutorial 


This repository aims to build a conceptually clean and minimal understanding of preference-based tuning for large language models, focusing on how reward signals, generation, and optimization interact.

The project is organized into three conceptual parts.


# Learning Comparison

| Algorithm | Training Objective | Reward Model | Computational Complexity |
|-----------|-------------------|--------------|-------------------------|
| **SFT** | Standard language modeling loss on chosen responses:<br>`L_SFT = -log π_θ(y_w | x)` | Not required | Low: Single forward pass per sample |
| **DPO** | Direct preference optimization:<br>`L_DPO = -log σ(β[(log π_θ(y_w) - log π_θ(y_l)) - (log π_ref(y_w) - log π_ref(y_l))])`<br>where σ is sigmoid, β is temperature, y_w is chosen, y_l is rejected | Not required | Medium: Requires forward passes for both policy and reference model |
| **RM** | Reward model training with ranking loss:<br>`L_RM = -log σ(r_φ(y_w) - r_φ(y_l))`<br>where r_φ is the reward model | Required (trained separately) | Low: Single forward pass per sample pair |
| **PPO** | Clipped PPO objective with KL penalty:<br>`L_PPO = -min(ρ_t A_t, clip(ρ_t, 1-ε, 1+ε) A_t) - λ_KL KL(π_θ || π_ref)`<br>where ρ_t = π_θ(a_t\|s_t) / π_old(a_t\|s_t) is importance ratio, A_t is advantage | Required | High: Requires online generation, multiple forward passes per update, and reward model inference |
| **GRPO** | Group relative policy optimization:<br>`L_GRPO = -(1/|P|) Σ_(i,j)∈P log σ(β[(log π_θ(y_i) - log π_θ(y_j)) - (log π_ref(y_i) - log π_ref(y_j))])`<br>where P = {(i,j) : r(y_i) > r(y_j)} is the set of preference pairs | Optional | Medium-High: Requires generating multiple responses per prompt and computing pairwise comparisons |