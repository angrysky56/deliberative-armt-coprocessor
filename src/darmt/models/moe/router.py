"""
Router for Mixture of Experts with top-k gating and load balancing.

The router determines which experts should process each token using
a learned gating function. Includes auxiliary load balancing loss to
prevent expert collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """
    Top-K routing mechanism with load balancing.
    
    Selects top-k experts for each token based on learned routing scores.
    Includes auxiliary loss to encourage balanced expert utilization.
    
    Based on:
    - Switch Transformer (top-1 routing)
    - Mixtral (top-2 routing)
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        jitter_noise: float = 0.01,
    ) -> None:
        """
        Initialize router.
        
        Args:
            hidden_size: Model hidden dimension
            num_experts: Total number of experts
            top_k: Number of experts to activate per token
            jitter_noise: Noise std for training stability (Mixtral uses 0.01)
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        
        # Routing layer: maps hidden states to expert scores
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Route tokens to top-k experts.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            routing_weights: Normalized weights for top-k experts [batch, seq_len, top_k]
            selected_experts: Indices of selected experts [batch, seq_len, top_k]
            auxiliary_loss: Dict containing load balancing loss
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Flatten for routing: [batch * seq_len, hidden_size]
        x_flat = x.view(-1, hidden_size)
        
        # Compute routing logits: [batch * seq_len, num_experts]
        logits = self.gate(x_flat)
        
        # Add jitter noise during training (Mixtral technique)
        if self.training and self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise
        
        # Get top-k experts per token
        routing_weights, selected_experts = self._top_k_gating(logits)
        
        # Compute load balancing auxiliary loss
        aux_loss = self._compute_load_balancing_loss(logits)
        
        # Reshape back: [batch, seq_len, top_k]
        routing_weights = routing_weights.view(batch_size, seq_len, self.top_k)
        selected_experts = selected_experts.view(batch_size, seq_len, self.top_k)
        
        return routing_weights, selected_experts, {"load_balance_loss": aux_loss}
    
    def _top_k_gating(
        self, 
        logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k experts and compute normalized routing weights.
        
        Args:
            logits: Router logits [num_tokens, num_experts]
            
        Returns:
            routing_weights: Softmax weights for top-k [num_tokens, top_k]
            selected_experts: Expert indices [num_tokens, top_k]
        """
        # Get top-k expert indices and their logits
        top_k_logits, selected_experts = torch.topk(logits, self.top_k, dim=-1)
        
        # Normalize top-k logits to get routing weights
        routing_weights = F.softmax(top_k_logits, dim=-1)
        
        return routing_weights, selected_experts
    
    def _compute_load_balancing_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Encourages balanced expert utilization by penalizing when some experts
        are used much more than others. Based on Switch Transformer formulation.
        
        Args:
            logits: Router logits [num_tokens, num_experts]
            
        Returns:
            Load balancing loss scalar
        """
        # Compute expert probabilities: [num_experts]
        probs = F.softmax(logits, dim=-1).mean(dim=0)
        
        # Compute fraction of tokens routed to each expert: [num_experts]
        # (based on top-1 routing for loss computation)
        top_expert = logits.argmax(dim=-1)
        expert_mask = F.one_hot(top_expert, num_classes=self.num_experts).float()
        fraction = expert_mask.mean(dim=0)
        
        # Load balance loss: num_experts * sum(probs * fraction)
        # This is minimized when all experts have equal load
        loss = self.num_experts * (probs * fraction).sum()
        
        return loss
