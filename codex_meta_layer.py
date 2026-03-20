import torch
import torch.nn as nn

class CodexMetaLayer(nn.Module):
    """Drop-in Codex constraint layer — add to any transformer"""
    def __init__(self):
        super().__init__()
        self.PCI = 1.0
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.alpha = nn.Parameter(torch.tensor(0.6))
    
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        true_C = 1 - torch.abs(H - self.PCI)
        S = (1 - true_C) ** 2
        perc_C = 1 - torch.abs(H - 0.6)          # example almost-PCI
        K = self.alpha * perc_C + (1 - self.alpha) * (1 - S)
        delta = self.gamma * (K - H)
        epsilon = torch.randn_like(H) * 0.03
        return torch.clamp(H + delta + epsilon, 0.0, 1.0)
