import torch
import torch.nn.functional as F

def get_logits(model, input_ids):
    outputs = model(input_ids=input_ids)
    return outputs.logits

def get_logprobs(model, input_ids, attention_mask=None):
    # This function assumes input_ids contains the full sequence (prompt + response)
    # or just response depending on usage. 
    # Based on for_ppo.py, it computes logprobs for the input_ids passed.
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits # [batch, seq, vocab]
    
    all_token_logprobs = F.log_softmax(logits, dim=-1)
    
    # We gather the logprob corresponding to the input tokens
    # input_ids: [batch, seq] -> [batch, seq, 1]
    token_logprobs = torch.gather(all_token_logprobs, 2, input_ids.unsqueeze(2)).squeeze(-1)
    
    return token_logprobs

def get_kl(logprobs_ref, logprobs, kl_ctl):
    kl = logprobs_ref - logprobs
    kl = kl * kl_ctl
    return kl

def get_reward_with_kl(logprobs_ref, logprobs, kl_ctl, reward):
    # This usually constructs a per-token reward
    # The 'reward' argument is usually the score for the entire sequence (scalar per batch item)
    # We add it to the last token's KL penalty.
    
    kl = get_kl(logprobs_ref, logprobs, kl_ctl)
    
    # Add task reward to the last token 
    # Assumes reward is [batch, 1] or [batch] and kl is [batch, seq]
    # We need to make sure we add it to the correct position (end of sequence)
    # For simplicity in this demo where we might treat fixed lengths:
    
    # Make a copy to avoid in-place modification issues if any
    kl_reward = kl.clone()
    
    # If reward is [batch, 1], flatten to [batch]
    if reward.dim() == 2:
        reward = reward.squeeze(1)
        
    kl_reward[:, -1] += reward
    
    return kl_reward

def get_gae(rewards, values_old, gamma, lam):
    # rewards: [batch, seq] (including KL)
    # values_old: [batch, seq] 
    
    # Note: GAE needs V(t+1). If values_old matches rewards length, we assume V(last+1) = 0
    
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgae = 0
    
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            nextvalues = 0.0
        else:
            nextvalues = values_old[:, t + 1]
            
        delta = rewards[:, t] + gamma * nextvalues - values_old[:, t]
        lastgae = delta + gamma * lam * lastgae
        advantages[:, t] = lastgae
        
    return advantages

def masked_mean(values, mask, axis=None):
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def get_value_loss(advantages, values, values_old, mask, cliprange_value):
    # Value Loss (Clipped)
    returns = values_old + advantages
    values_clipped = torch.clamp(values, values_old - cliprange_value, values_old + cliprange_value)
    
    vf_losses1 = torch.square(values_clipped - returns)
    vf_losses2 = torch.square(values - returns)
    vf_loss_max = torch.max(vf_losses1, vf_losses2)
    
    vf_loss = 0.5 * masked_mean(vf_loss_max, mask)
    return vf_loss

def get_policy_loss(advantages, logprobs, logprobs_old, mask, cliprange):
    # Ratio = exp(new - old)
    ratio = torch.exp(logprobs - logprobs_old)
    
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    
    pg_loss_max = torch.max(pg_losses1, pg_losses2)
    pg_loss = masked_mean(pg_loss_max, mask)
    
    return pg_loss
