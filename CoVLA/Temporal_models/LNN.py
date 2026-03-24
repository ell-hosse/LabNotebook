import torch
import torch.nn as nn


class LiquidCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.x2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.x2a = nn.Linear(input_dim, hidden_dim)
        self.h2a = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.x2h(x_t) + self.h2h(h_t))

        alpha = torch.sigmoid(self.x2a(x_t) + self.h2a(h_t))

        h_next = (1.0 - alpha) * h_t + alpha * candidate
        h_next = self.dropout(h_next)
        h_next = self.norm(h_next)

        return h_next


class LNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        traj_points: int = 10,
        coord_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.traj_points = traj_points
        self.coord_dim = coord_dim
        self.out_dim = traj_points * coord_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cell = LiquidCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {D}")

        x = self.input_proj(x)

        if h0 is None:
            h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h_t = h0

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]
            h_t = self.cell(x_t, h_t)

            y_t = self.head(h_t)
            y_t = y_t.view(B, self.traj_points, self.coord_dim)

            outputs.append(y_t)

        pred = torch.stack(outputs, dim=1)
        return pred

