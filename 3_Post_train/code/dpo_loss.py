import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DPOLoss(nn.Module):
    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        """
        初始化 DPO 损失函数组件。
        
        Args:
            beta (float): 温度参数，控制对参考模型的偏离程度。Beta 越大，对偏离的惩罚越大。
            label_smoothing (float): 标签平滑参数 (0.0 到 1.0)。用于防止模型过度自信，增强鲁棒性。
            ipo (bool): 是否使用 IPO (Identity Preference Optimization) 损失代替 DPO 损失。
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(self, 
                policy_chosen_logps: torch.Tensor,
                policy_rejected_logps: torch.Tensor,
                reference_chosen_logps: torch.Tensor,
                reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 DPO 损失。

        Args:
            policy_chosen_logps: 策略模型对"胜出"样本的 log 概率。
            policy_rejected_logps: 策略模型对"失败"样本的 log 概率。
            reference_chosen_logps: 参考模型对"胜出"样本的 log 概率。
            reference_rejected_logps: 参考模型对"失败"样本的 log 概率。
        
        Returns:
            Tuple[loss, chosen_rewards, rejected_rewards]
        """
        # 计算策略模型和参考模型的 log ratios (对数几率比)
        # log(p_chosen / p_rejected) = log(p_chosen) - log(p_rejected)
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        # Logits 是策略模型相对于参考模型的优势差异
        # logits = log(pi_chosen/ref_chosen) - log(pi_rejected/ref_rejected)
        logits = pi_logratios - ref_logratios

        if self.ipo:
            # IPO 损失：直接最小化 logits 与正则化项的均方误差
            # (logits - 1/(2*beta))^2
            losses = (logits - 1/(2 * self.beta)) ** 2
        else:
            # DPO 损失：带有标签平滑（Label Smoothing）
            # 如果没有标签平滑，模型会拼命拉大好坏回答的差距，导致过拟合。
            # 加上平滑后（例如 label_smoothing=0.1），模型就不会过于自信，训练更加鲁棒
            # 这实际上是一个带平滑的二元交叉熵损失 (BCE Loss)
            # 目标标签不是纯粹的 1，而是 1 - label_smoothing
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            
        loss = losses.mean()
        
        # 计算隐式奖励 (Implicit Rewards) 用于监控
        # reward = beta * (log(pi(y|x)) - log(ref(y|x)))
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        return loss, chosen_rewards, rejected_rewards