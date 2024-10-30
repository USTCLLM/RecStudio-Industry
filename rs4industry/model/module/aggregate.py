import torch


__all__ = ["AverageAggregator"]

class AverageAggregator(torch.nn.Module):
    def __init__(self, dim=1, *args, **kwargs):
        """ Average sequence aggregator.

        Args:
            dim (int, optional): Dimension to average over. Defaults to 1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq (torch.Tensor): [batch_size, length, hidden_dim]
        """
        return torch.mean(seq, dim=self.dim)
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"