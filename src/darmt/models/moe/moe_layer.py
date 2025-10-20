"""
Sparse Mixture of Experts layer.

Combines router and experts to create a sparse MoE layer that can
replace standard FFN layers in transformers.
"""

import torch
import torch.nn as nn

from darmt.models.moe.expert import Expert
from darmt.models.moe.router import TopKRouter


class MoELayer(nn.Module):
    """
    Sparse MoE layer with top-k routing.
    
    This layer:
    1. Uses router to select top-k experts per token
    2. Processes tokens through selected experts only (sparse activation)
    3. Combines expert outputs using routing weights
    4. Returns auxiliary load balancing loss
    
    Can replace standard FFN layers in transformers.
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_loss_coef: float = 0.01,
    ) -> None:
        """
        Initialize MoE layer.
        
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: Expert FFN intermediate dimension
            num_experts: Total number of experts
            top_k: Number of experts to activate per token
            dropout: Dropout probability in experts
            load_balance_loss_coef: Weight for load balancing auxiliary loss
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_loss_coef = load_balance_loss_coef
        
        # Router
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            output: Combined expert outputs [batch, seq_len, hidden_size]
            aux_loss_dict: Dictionary with auxiliary losses
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Route tokens to experts
        routing_weights, selected_experts, aux_loss = self.router(x)
        
        # Flatten for expert processing: [batch * seq_len, hidden_size]
        x_flat = x.view(-1, hidden_size)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process tokens through selected experts (sparse)
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts.view(-1, self.top_k) == i).any(dim=-1)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]
                
                # Process through expert
                expert_output = self.experts[i](expert_input)
                
                # Get routing weights for this expert's tokens
                expert_weights = torch.zeros(x_flat.size(0), device=x.device)
                for k in range(self.top_k):
                    mask = (selected_experts.view(-1, self.top_k)[:, k] == i)
                    expert_weights[mask] = routing_weights.view(-1, self.top_k)[mask, k]
                
                # Add weighted expert output
                output[expert_mask] += expert_output * expert_weights[expert_mask].unsqueeze(-1)
        
        # Reshape back: [batch, seq_len, hidden_size]
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Scale auxiliary loss
        scaled_aux_loss = {
            "load_balance_loss": aux_loss["load_balance_loss"] * self.load_balance_loss_coef
        }
        
        return output, scaled_aux_loss
