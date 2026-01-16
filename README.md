# LLM RL Tutorial 


This repository aims to build a conceptually clean and minimal understanding of preference-based tuning for large language models, focusing on how reward signals, generation, and optimization interact.

The project is organized into three conceptual parts.


# Learning Comparison

| Algorithm | Training Objective | Reward Model | Computational Complexity |
|-----------|-------------------|--------------|-------------------------|
| **SFT** | Standard language modeling loss on chosen responses:<br>$\mathcal{L}_{\text{SFT}} = -\log \pi_\theta(y_w \mid x)$ | Not required | Low: Single forward pass per sample |
| **DPO** | Direct preference optimization:<br>$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[(\log \pi_\theta(y_w) - \log \pi_\theta(y_l)) - (\log \pi_{\text{ref}}(y_w) - \log \pi_{\text{ref}}(y_l))\right]\right)$<br>where $\sigma$ is sigmoid, $\beta$ is temperature, $y_w$ is chosen, $y_l$ is rejected | Not required | Medium: Requires forward passes for both policy and reference model |
| **RM** | Reward model training with ranking loss:<br>$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(y_w) - r_\phi(y_l))$<br>where $r_\phi$ is the reward model | Required (trained separately) | Low: Single forward pass per sample pair |
| **PPO** | Clipped PPO objective with KL penalty:<br>$\mathcal{L}_{\text{PPO}} = -\min\left(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right) - \lambda_{\text{KL}} \text{KL}(\pi_\theta \| \pi_{\text{ref}})$<br>where $\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$ is importance ratio, $A_t$ is advantage | Required | High: Requires online generation, multiple forward passes per update, and reward model inference |
| **GRPO** | Group relative policy optimization:<br>$\mathcal{L}_{\text{GRPO}} = -\frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \log \sigma\left(\beta \left[(\log \pi_\theta(y_i) - \log \pi_\theta(y_j)) - (\log \pi_{\text{ref}}(y_i) - \log \pi_{\text{ref}}(y_j))\right]\right)$<br>where $\mathcal{P} = \{(i,j) : r(y_i) > r(y_j)\}$ is the set of preference pairs | Optional | Medium-High: Requires generating multiple responses per prompt and computing pairwise comparisons |