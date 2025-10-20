"""
Expert FFN module for Mixture of Experts.

Each expert is a standard feed-forward network that can specialize
in processing specific token-context patterns.
"""

import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    Single expert feed-forward network.
    
    This is a standard 2-layer FFN with GELU activation,
    identical to what you'd find in a transformer FFN layer.
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize expert FFN.
        
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension (typically 4x hidden)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Upproject + activation
        hidden = self.activation(self.fc1(x))
        hidden = self.dropout(hidden)
        
        # Downproject
        output = self.fc2(hidden)
        
        return output
