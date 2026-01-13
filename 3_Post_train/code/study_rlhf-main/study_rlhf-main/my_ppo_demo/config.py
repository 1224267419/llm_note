import torch

class PPOConfig:
    def __init__(self):
        # Model Parameters
        self.vocab_size = 50257  # For the purpose of the demo
        self.hidden_size = 128
        self.intermediate_size = 256
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.max_new_tokens = 5
        self.length_x = 5 # prompt length
        
        # Training Parameters
        self.ppo_epochs = 20
        self.mini_batch_size = 5
        self.epochs = 5
        self.learning_rate = 1e-4 # Added learning rate
        self.batch_size = 3
        
        # PPO Hyperparameters
        self.kl_ctl = 0.1
        self.vf_coef = 0.1
        self.lam = 0.9
        self.gamma = 0.9
        self.cliprange_value = 0.2
        self.cliprange = 0.2 # Added cliprange for policy loss
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return (f'ppo_epochs:{self.ppo_epochs}\n'
                f'mini_batch_size:{self.mini_batch_size}\n'
                f'epochs:{self.epochs}\n'
                f'kl_ctl:{self.kl_ctl}\n'
                f'device:{self.device}')
