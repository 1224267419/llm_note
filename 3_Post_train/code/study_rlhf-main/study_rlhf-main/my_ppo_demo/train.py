import torch
from torch import optim
from config import PPOConfig
from models import PPOModels
from utils import (
    get_logprobs, 
    get_reward_with_kl, 
    get_gae, 
    get_policy_loss, 
    get_value_loss
)

def train():
    # 1. Initialization
    config = PPOConfig()
    print(f"Initializing PPO with config:\n{config}")
    
    ppo_models = PPOModels(config)
    
    # Optimizers
    actor_optimizer = optim.AdamW(ppo_models.actor.parameters(), lr=config.learning_rate)
    critic_optimizer = optim.AdamW(ppo_models.critic.parameters(), lr=config.learning_rate)
    
    # Loss history
    all_losses = {'total': [], 'pg': [], 'vf': []}
    
    # Dummy Data Generator (Prompt)
    # in real scenario, this comes from a dataset
    def get_batch_data(batch_size, vocab_size, length_x):
        return torch.randint(0, vocab_size, (batch_size, length_x)).to(config.device)

    # 2. Main Loop
    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        
        # --- Rollout Phase ---
        # 2.1 Sample prompts
        prompts = get_batch_data(config.batch_size, config.vocab_size, config.length_x)
        
        # 2.2 Generate Responses (Actor)
        ppo_models.actor.eval()
        with torch.no_grad():
            # Attention mask for generation (all 1s for prompt)
            prompt_mask = torch.ones_like(prompts)
            
            # Generate
            # max_length = prompt_len + new_tokens
            max_length = config.length_x + config.max_new_tokens
            
            sequences = ppo_models.actor.generate(
                prompts, 
                max_new_tokens=config.max_new_tokens,
                pad_token_id=ppo_models.gpt_config.eos_token_id
            )
            
        # Create masks
        # We want to train on the response part only
        # sequence: [prompt, response]
        # mask should be 0 for prompt, 1 for response
        
        attention_mask = torch.ones_like(sequences)
        attention_mask[:, :config.length_x] = 0 # Mask out prompt
        
        print(f"Sample Sequence: {sequences[0].tolist()}")
        
        # 2.3 Compute Old Logprobs, Values, Rewards
        # We need gradients for nothing here, this is data collection
        with torch.no_grad():
             # Logprobs from Reference Model
             logprobs_ref = get_logprobs(ppo_models.ref, sequences)
             
             # Logprobs from Actor Model (Old Policy)
             logprobs_old = get_logprobs(ppo_models.actor, sequences)
             
             # Values from Critic (Old Value)
             values_old = ppo_models.critic(sequences)
             
             # Rewards from Reward Model
             # Note: Typically RM takes (input_ids, attention_mask)
             # We might want to pass the full mask to RM, but usually RM just scores the sequence.
             # Here we treat the full sequence as valid input for RM.
             rewards_raw = ppo_models.rm(sequences)
             
        # 2.4 Compute Rewards with KL Penalty
        # rewards_raw is [batch, 1], logprobs are [batch, seq]
        # We compute a "per-token" reward where only the last token has the task reward
        # and all tokens have KL penalty.
        rewards = get_reward_with_kl(logprobs_ref, logprobs_old, config.kl_ctl, rewards_raw)
        
        # 2.5 Compute GAE (Advantages) and Returns
        advantages = get_gae(rewards, values_old, config.gamma, config.lam)
        
        # Normalize Advantages (Standard PPO practice for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # --- Update Phase ---
        ppo_models.actor.train()
        ppo_models.critic.train()
        
        # Re-compute masking for loss: 
        # utils.masked_mean uses the mask to compute average. 
        # We only want to average over response tokens.
        loss_mask = attention_mask
        
        for ppo_epoch in range(config.ppo_epochs):
             # In a real implementation, we would iterate over mini-batches here.
             # For this simple demo/tutorial, we'll just do a single full-batch update 
             # or a simple slice if we wanted to match mini_batch_size exactly.
             # Let's iterate full batch for simplicity or strict mini-batches?
             # Let's do a simple shuffle and mini-batch loop to be reasonably complete
             
             indices = torch.randperm(config.batch_size)
             for i in range(0, config.batch_size, config.mini_batch_size):
                 batch_indices = indices[i:i + config.mini_batch_size]
                 if len(batch_indices) == 0: continue
                 
                 # Slice data
                 b_sequences = sequences[batch_indices]
                 b_mask = loss_mask[batch_indices]
                 b_logprobs_old = logprobs_old[batch_indices]
                 b_advantages = advantages[batch_indices]
                 b_values_old = values_old[batch_indices]
                 
                 # Forward pass (New Policy)
                 b_logprobs_new = get_logprobs(ppo_models.actor, b_sequences)
                 
                 # Forward pass (New Value)
                 b_values_new = ppo_models.critic(b_sequences)
                 
                 # Calculate Losses
                 pg_loss = get_policy_loss(
                     b_advantages, 
                     b_logprobs_new, 
                     b_logprobs_old, 
                     b_mask, 
                     config.cliprange
                 )
                 
                 vf_loss = get_value_loss(
                     b_advantages, 
                     b_values_new, 
                     b_values_old, 
                     b_mask, 
                     config.cliprange_value
                 )
                 
                 loss = pg_loss + config.vf_coef * vf_loss
                 
                 # Backward
                 actor_optimizer.zero_grad()
                 critic_optimizer.zero_grad()
                 loss.backward()
                 actor_optimizer.step()
                 critic_optimizer.step()
                 
             print(f"  PPO Epoch {ppo_epoch+1}: Total Loss = {loss.item():.4f} (PG: {pg_loss.item():.4f}, VF: {vf_loss.item():.4f})")
             
             # Record losses
             all_losses['total'].append(loss.item())
             all_losses['pg'].append(pg_loss.item())
             all_losses['vf'].append(vf_loss.item())

    print("Training Completed.")
    
    # Plotting
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses['total'], label='Total Loss')
    plt.plot(all_losses['pg'], label='Policy Gradient Loss')
    plt.plot(all_losses['vf'], label='Value Function Loss')
    plt.xlabel('Update Steps')
    plt.ylabel('Loss')
    plt.title('PPO Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('./loss_curve.png')
    print("Loss curve saved to 'loss_curve.png'")

if __name__ == "__main__":
    train()
