# architecture.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from efficient_kan import KANLinear
except ImportError as e:
    raise ImportError(
        "Please install `efficient-kan` first: pip install efficient-kan"
    ) from e


def _make_kan(
    in_dim: int,
    out_dim: int,
    grid_size: int = 8,
    spline_order: int = 3,
    scale_noise: float = 0.1,
    scale_base: float = 1.0,
    scale_spline: float = 1.0,
    enable_standalone_scale_spline: bool = True,
    base_activation=nn.SiLU,
    grid_eps: float = 0.02,
    grid_range: Tuple[float, float] = (-1.0, 1.0),
) -> KANLinear:
    return KANLinear(
        in_features=in_dim,
        out_features=out_dim,
        grid_size=grid_size,
        spline_order=spline_order,
        scale_noise=scale_noise,
        scale_base=scale_base,
        scale_spline=scale_spline,
        enable_standalone_scale_spline=enable_standalone_scale_spline,
        base_activation=base_activation,
        grid_eps=grid_eps,
        grid_range=list(grid_range),
    )


class KANBlock(nn.Sequential):

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        grid_size: int = 8,
        spline_order: int = 3,
    ):
        layers: List[nn.Module] = []
        dims = [in_dim] + list(hidden_dims)
        for d0, d1 in zip(dims[:-1], dims[1:]):
            layers.append(_make_kan(d0, d1, grid_size=grid_size, spline_order=spline_order))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        if out_dim is not None:
            layers.append(_make_kan(dims[-1], out_dim, grid_size=grid_size, spline_order=spline_order))
        super().__init__(*layers)

    def regularization_loss(self, reg_activation: float = 1.0, reg_entropy: float = 1.0) -> torch.Tensor:
        reg = 0.0
        for m in self.modules():
            if isinstance(m, KANLinear):
                reg = reg + m.regularization_loss(
                    regularize_activation=reg_activation,
                    regularize_entropy=reg_entropy,
                )
        return reg if isinstance(reg, torch.Tensor) else torch.tensor(reg)


class TemporalKernelAttentionKAN(nn.Module):

    def __init__(
        self,
        n_kernels: int = 4,
        d_att: int = 64,
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.0,
        normalize_weights: bool = True,
    ):
        super().__init__()
        self.n_k = n_kernels
        self.normalize = normalize_weights
        self.pi_logits = nn.Parameter(torch.zeros(n_kernels))
        self.lam_raw = nn.Parameter(torch.zeros(n_kernels))
        self.summarize = KANBlock(1, hidden_dims=(d_att,), out_dim=d_att, dropout=dropout, grid_size=grid_size, spline_order=spline_order)
        self.out_head = KANBlock(d_att, hidden_dims=(d_att,), out_dim=1, dropout=dropout, grid_size=grid_size, spline_order=spline_order)

    def forward(self, y_lags: torch.Tensor, dt_lags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = y_lags.shape
        pi = F.softmax(self.pi_logits, dim=-1)                      
        lam = F.softplus(self.lam_raw) + 1e-6                       
        kernel = torch.exp(-lam.view(1, 1, -1) * dt_lags.unsqueeze(-1))  
        w_lags = torch.sum(pi.view(1, 1, -1) * kernel, dim=-1)      
        if self.normalize:
            w_lags = w_lags / (w_lags.sum(dim=1, keepdim=True) + 1e-8)
        s = torch.sum(w_lags * y_lags, dim=1, keepdim=True)         
        z_att = self.summarize(s)                                   
        e_att = self.out_head(z_att)
        return e_att, w_lags, z_att


class FixedBranchKAN(nn.Module):
    def __init__(self, d_fix: int, d_latent: int = 128, grid_size: int = 8, spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.enc = KANBlock(d_fix, hidden_dims=(256, 128), out_dim=d_latent, dropout=dropout, grid_size=grid_size, spline_order=spline_order)
        self.head = KANBlock(d_latent, hidden_dims=(64,), out_dim=1, dropout=dropout, grid_size=grid_size, spline_order=spline_order)

    def forward(self, X_fix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(X_fix)
        e = self.head(z)
        return e, z

    def regularization_loss(self) -> torch.Tensor:
        return self.enc.regularization_loss() + self.head.regularization_loss()


class RandEncoderKAN(nn.Module):
    def __init__(self, d_zrand: int, d_latent: int = 128, grid_size: int = 8, spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.enc = KANBlock(d_zrand, hidden_dims=(256, 128), out_dim=d_latent, dropout=dropout, grid_size=grid_size, spline_order=spline_order)

    def forward(self, Zrand: torch.Tensor) -> torch.Tensor:
        return self.enc(Zrand)

    def regularization_loss(self) -> torch.Tensor:
        return self.enc.regularization_loss()


class TCEncoderKAN(nn.Module):
    def __init__(self, d_tc: int, d_latent: int = 128, grid_size: int = 8, spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.enc = KANBlock(d_tc, hidden_dims=(256, 128), out_dim=d_latent, dropout=dropout, grid_size=grid_size, spline_order=spline_order)

    def forward(self, TC: torch.Tensor) -> torch.Tensor:
        return self.enc(TC)

    def regularization_loss(self) -> torch.Tensor:
        return self.enc.regularization_loss()


class RandomHeadKAN(nn.Module):
    def __init__(self, d_latent: int = 128, grid_size: int = 8, spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.head = KANBlock(d_latent, hidden_dims=(64,), out_dim=1, dropout=dropout, grid_size=grid_size, spline_order=spline_order)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

    def regularization_loss(self) -> torch.Tensor:
        return self.head.regularization_loss()


class FiLMFromTC(nn.Module):

    def __init__(self, d_latent: int = 128, grid_size: int = 8, spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.gamma = KANBlock(d_latent, hidden_dims=(64,), out_dim=d_latent, dropout=dropout, grid_size=grid_size, spline_order=spline_order)
        self.beta = KANBlock(d_latent, hidden_dims=(64,), out_dim=d_latent, dropout=dropout, grid_size=grid_size, spline_order=spline_order)

    def forward(self, e_tc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gamma(e_tc), self.beta(e_tc)

    def regularization_loss(self) -> torch.Tensor:
        return self.gamma.regularization_loss() + self.beta.regularization_loss()


@dataclass
class KANDefaults:
    d_fix_latent: int = 256
    d_rand_latent: int = 256
    d_att: int = 128
    n_kernels: int = 8
    grid_size: int = 8
    spline_order: int = 3
    dropout: float = 0.0
    reg_activation: float = 1.0
    reg_entropy: float = 1.0


class KANAdditiveMixed(nn.Module):

    def __init__(
        self,
        y_dim: int,
        d_fix: int,
        d_tc: int,
        d_zrand: int,
        defaults: KANDefaults = KANDefaults(),
    ):
        super().__init__()
        self.y_dim = int(y_dim)
        self.d_fix = int(d_fix)
        self.d_tc = int(d_tc)
        self.d_zr = int(d_zrand)
        self.cfg = defaults

 
        self.att = nn.ModuleList([
            TemporalKernelAttentionKAN(
                n_kernels=self.cfg.n_kernels,
                d_att=self.cfg.d_att,
                grid_size=self.cfg.grid_size,
                spline_order=self.cfg.spline_order,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.y_dim)
        ])


        self.fix = nn.ModuleList([
            FixedBranchKAN(
                d_fix=self.d_fix,
                d_latent=self.cfg.d_fix_latent,
                grid_size=self.cfg.grid_size,
                spline_order=self.cfg.spline_order,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.y_dim)
        ])

 
        self.rand_enc = nn.ModuleList([
            RandEncoderKAN(
                d_zrand=max(1, self.d_zr),  
                d_latent=self.cfg.d_rand_latent,
                grid_size=self.cfg.grid_size,
                spline_order=self.cfg.spline_order,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.y_dim)
        ])
        self.tc_enc = nn.ModuleList([
            TCEncoderKAN(
                d_tc=max(1, self.d_tc),
                d_latent=self.cfg.d_rand_latent,
                grid_size=self.cfg.grid_size,
                spline_order=self.cfg.spline_order,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.y_dim)
        ])
        self.film = nn.ModuleList([
            FiLMFromTC(
                d_latent=self.cfg.d_rand_latent,
                grid_size=self.cfg.grid_size,
                spline_order=self.cfg.spline_order,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.y_dim)
        ])
        self.rand_head = nn.ModuleList([
            RandomHeadKAN(
                d_latent=self.cfg.d_rand_latent,
                grid_size=self.cfg.grid_size,
                spline_order=self.cfg.spline_order,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.y_dim)
        ])

    @classmethod
    def from_dims(
        cls,
        y_dim: int,
        d_fix: int,
        d_tc: int,
        d_zrand: int,
        **overrides: Any,
    ) -> "KANAdditiveMixed":
        defaults = KANDefaults(**overrides) if overrides else KANDefaults()
        return cls(y_dim=y_dim, d_fix=d_fix, d_tc=d_tc, d_zrand=d_zrand, defaults=defaults)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        X_fix: torch.Tensor,
        TC: Optional[torch.Tensor],
        Zrand: Optional[torch.Tensor],
        y_lags: Optional[torch.Tensor],
        dt_lags: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = X_fix.device
        B = X_fix.size(0)

        if y_lags is None:
            y_lags = torch.zeros(B, 1, device=device, dtype=X_fix.dtype)
        if dt_lags is None:
            dt_lags = torch.zeros(B, 1, device=device, dtype=X_fix.dtype)
        if TC is None or TC.size(1) == 0:
            TC = torch.zeros(B, max(1, self.d_tc), device=device, dtype=X_fix.dtype)
        if Zrand is None or Zrand.size(1) == 0:
            Zrand = torch.zeros(B, max(1, self.d_zr), device=device, dtype=X_fix.dtype)

        if y_lags.dim() == 3:
            assert y_lags.size(2) == self.y_dim
            y_lags_list = [y_lags[:, :, j] for j in range(self.y_dim)]
        else:
            y_lags_list = [y_lags for _ in range(self.y_dim)]

        e_att_all, e_fix_all, e_rand_all = [], [], []
        z_att_list, z_fix_list, z_rand_list, z_tc_list, z_film_list = [], [], [], [], []
        w_lags_list, gamma_list, beta_list = [], [], []

        for j in range(self.y_dim):
            e_att_j, w_lags_j, z_att_j = self.att[j](y_lags_list[j], dt_lags)

            e_fix_j, z_fix_j = self.fix[j](X_fix)

            z_rand_j = self.rand_enc[j](Zrand)
            z_tc_j = self.tc_enc[j](TC)
            gamma_j, beta_j = self.film[j](z_tc_j)
            z_tilde_j = gamma_j * z_rand_j + beta_j
            e_rand_j = self.rand_head[j](z_tilde_j)

            e_att_all.append(e_att_j)
            e_fix_all.append(e_fix_j)
            e_rand_all.append(e_rand_j)

            z_att_list.append(z_att_j)
            z_fix_list.append(z_fix_j)
            z_rand_list.append(z_rand_j)
            z_tc_list.append(z_tc_j)
            z_film_list.append(z_tilde_j)
            w_lags_list.append(w_lags_j)
            gamma_list.append(gamma_j)
            beta_list.append(beta_j)

        e_att = torch.cat(e_att_all, dim=1)
        e_fix = torch.cat(e_fix_all, dim=1)
        e_rand = torch.cat(e_rand_all, dim=1)
        logits = e_att + e_fix + e_rand

        parts = dict(
            e_att=e_att,
            e_fix=e_fix,
            e_rand=e_rand,
            z_att_list=z_att_list,
            z_fix_list=z_fix_list,
            z_rand_list=z_rand_list,
            z_tc_list=z_tc_list,
            z_rand_film_list=z_film_list,
            w_lags_list=w_lags_list,
            film_gamma_list=gamma_list,
            film_beta_list=beta_list,
        )
        return logits, parts

    def regularization_loss(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for fb in self.fix:
            reg = reg + fb.regularization_loss()
        for j in range(self.y_dim):
            reg = reg + self.att[j].summarize.regularization_loss() + self.att[j].out_head.regularization_loss()
            reg = reg + self.rand_enc[j].regularization_loss() + self.tc_enc[j].regularization_loss()
            reg = reg + self.film[j].regularization_loss() + self.rand_head[j].regularization_loss()
        return reg
