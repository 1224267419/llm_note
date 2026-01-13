import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

class GPTRewardModel(nn.Module):
    def __init__(self, gpt_model, reward_head):
        super(GPTRewardModel, self).__init__()
        self.gpt_model = gpt_model
        self.reward_head = reward_head
        
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
             attention_mask = torch.ones_like(input_ids)
             
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # In this demo, we can just use the last token's representation for the reward 
        # But realistically we might want to extract the specific token that corresponds to the end of the sequence.
        # Following the notebook demo logic:
        batch_size = input_ids.shape[0]
        
        # Calculate sequence length (index of the last valid token)
        # -1 because indices start at 0
        if attention_mask is not None:
             sequence_length = attention_mask.sum(dim=1).long() - 1
        else:
             sequence_length = torch.full((batch_size,), input_ids.shape[1] - 1, device=input_ids.device).long()

        batch_indices = torch.arange(batch_size, device=input_ids.device).long()
        
        # Select the last hidden state for valid tokens
        selected_hidden_state = last_hidden_state[batch_indices, sequence_length]
        
        rewards = self.reward_head(selected_hidden_state)
        return rewards

class GPTValueModel(nn.Module):
    def __init__(self, gpt_model, value_head):
        super().__init__()
        self.gpt_model = gpt_model
        self.value_head = value_head
        
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
             attention_mask = torch.ones_like(input_ids)
             
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Value is calculated for every token
        values = self.value_head(last_hidden_state).squeeze(-1) # [batch_size, seq_len]
        return values

class PPOModels:
    def __init__(self, config):
        self.config = config
        
        # Initialize GPT-2 Configuration
        self.gpt_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=config.hidden_size,
            n_inner=config.intermediate_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads
        )
        
        # Actor Model (Policy)
        self.actor = GPT2LMHeadModel(self.gpt_config)
        self.actor.to(config.device)
        
        # Reference Model (Frozen Copy of Actor)
        self.ref = GPT2LMHeadModel(self.gpt_config)
        self.ref.load_state_dict(self.actor.state_dict())
        self.ref.to(config.device)
        self.ref.eval() # Always eval
        for param in self.ref.parameters():
            param.requires_grad = False
            
        # Reward Model
        # We use a separate model for reward in this demo, initialized similarly
        gpt_for_rm = GPT2LMHeadModel(self.gpt_config)
        self.rm = GPTRewardModel(gpt_for_rm, nn.Linear(config.hidden_size, 1))
        self.rm.to(config.device)
        
        # Critic Model (Value)
        # Similar architecture but different head
        gpt_for_vm = GPT2LMHeadModel(self.gpt_config)
        self.critic = GPTValueModel(gpt_for_vm, nn.Linear(config.hidden_size, 1))
        self.critic.to(config.device)
