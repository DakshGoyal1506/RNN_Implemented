"""
A simple RNN implementation using PyTorch.
This module defines a SimpleRNN class that can be used for sequence modeling tasks.
"""

import torch
from torch import nn

class SimpleRNN(nn.Module):
    """A simple RNN implementation using PyTorch.

    Attributes:
        input_size (int): The size of the input features.
        hidden_units (int): The number of hidden units in the RNN.
        output_size (int): The size of the output features.
    """

    def __init__(self, 
                 input_size: int,
                 hidden_units: int,
                 output_size: int):
        """Initializes the SimpleRNN model.

        Args:
            input_size (int): The size of the input features.
            hidden_units (int): The number of hidden units in the RNN.
            output_size (int): The size of the output features.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_units, bias=False)
        self.h2h = nn.Linear(hidden_units, hidden_units)
        self.h2o = nn.Linear(hidden_units, output_size)
    
    def forward(self,
                x: torch.Tensor,
                hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the RNN.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_size).
            hidden_state (torch.Tensor): The hidden state tensor of shape (batch_size, hidden_units).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The output tensor of shape (batch_size, output_size).
                - The updated hidden state tensor of shape (batch_size, hidden_units).
        """
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)
        if self.h2o.out_features > 0:
            output = self.h2o(hidden_state)
        else:
            output = None
        return output, hidden_state
    
    def init_zero_hidden(self,
                         batch_size: int = 1) -> torch.Tensor:
        """Initializes the hidden state with zeros.

        Args:
            batch_size (int, optional): The batch size. Defaults to 1.

        Returns:
            torch.Tensor: A tensor of zeros with shape (batch_size, hidden_units).
        """
        return torch.zeros(batch_size, self.hidden_units, requires_grad=False)
