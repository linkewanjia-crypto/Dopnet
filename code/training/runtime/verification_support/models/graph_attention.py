"""Sparse axial graph primitives used by DOP-Net.

The attention logits avoid gathering the full projected feature vector of every
neighbour. Because the GAT logit is additive,

    e_{ij,h} = LeakyReLU( a_src[h] . q_i[h] + a_dst[h] . q_j[h] ),

the two inner products can be evaluated *per node* first and only the resulting
scalars gathered.  That replaces a ``[B_ax,N,K,H,Dh]`` intermediate on the logit
path by a ``[B_ax,N,K,H]`` one -- a ``d_head`` fold reduction (12x at the
shipped width) -- and is mathematically identical.

The aggregation step still needs the neighbour values, which is unavoidable for
gather-based sparse attention; that tensor is what
``model.grad_checkpoint: true`` in the config exists to keep out of the autograd
tape.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialSparseGraphBuilder(nn.Module):
    """Builds only candidate edges on one axis.

    ``radius=None`` gives full candidate connectivity along the selected axis,
    from which ``top_k`` neighbours are then selected by a sample-adaptive mix
    of a physical-distance prior and a learned cosine affinity.
    """

    def __init__(self, axis_size, sigma, radius=None, top_k=None):
        super().__init__()
        self.axis_size = int(axis_size)
        self.sigma = float(sigma)
        self.radius = None if radius is None else int(radius)

        if self.radius is None:
            centers = torch.arange(self.axis_size)[:, None]
            candidate_idx = torch.arange(self.axis_size)[None, :].expand(
                self.axis_size, self.axis_size
            ).clone()
            valid = torch.ones_like(candidate_idx, dtype=torch.bool)
            distance = (candidate_idx - centers).abs().float()
        else:
            offsets = torch.arange(-self.radius, self.radius + 1)
            centers = torch.arange(self.axis_size)[:, None]
            raw = centers + offsets[None, :]
            valid = (raw >= 0) & (raw < self.axis_size)
            candidate_idx = raw.clamp(0, self.axis_size - 1)
            distance = offsets.abs().float()[None, :].expand(self.axis_size, -1)

        physical = torch.exp(-distance / self.sigma) * valid.float()
        self.register_buffer("candidate_idx", candidate_idx.long())
        self.register_buffer("candidate_valid", valid)
        self.register_buffer("physical_prior", physical)

        max_candidates = int(candidate_idx.shape[1])
        if top_k is None:
            top_k = max_candidates
        self.top_k = min(int(top_k), max_candidates)

        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.25)))

    @property
    def max_candidates(self):
        return int(self.candidate_idx.shape[1])

    @property
    def valid_edge_count(self):
        return int(self.candidate_valid.sum().item())

    @property
    def edge_density(self):
        """Fraction of the axis each node actually attends to."""
        return float(self.top_k) / float(max(1, self.axis_size))

    def forward(self, nodes, lam):
        """
        nodes: [Bax, Naxis, D]
        lam:   [Bax] or [Bax,1,1]

        returns (neighbor_idx, prior, valid, learned, fused), each [Bax,Naxis,K].
        """
        B, N, _ = nodes.shape
        if N != self.axis_size:
            raise ValueError(
                f"Axis-size mismatch: builder expects {self.axis_size}, got {N}."
            )

        xn = F.normalize(nodes, dim=-1)
        idx = self.candidate_idx
        if self.radius is None:
            learned_cos = torch.matmul(xn, xn.transpose(-1, -2))
        else:
            nbr = xn[:, idx, :]
            learned_cos = (xn.unsqueeze(2) * nbr).sum(dim=-1)

        temp = self.log_temperature.exp().clamp(0.03, 3.0)
        learned = torch.sigmoid(learned_cos / temp)

        valid = self.candidate_valid.unsqueeze(0).expand(B, -1, -1)
        physical = self.physical_prior.unsqueeze(0).expand(B, -1, -1)

        lam = lam.reshape(B, 1, 1)
        fused = lam * physical + (1.0 - lam) * learned
        fused = fused.masked_fill(~valid, -1.0)

        if self.top_k < self.max_candidates:
            vals, order = torch.topk(fused, k=self.top_k, dim=-1, largest=True, sorted=False)
            idx_b = idx.unsqueeze(0).expand(B, -1, -1)
            idx_b = torch.gather(idx_b, 2, order)
            valid_b = torch.gather(valid, 2, order)
            learned_b = torch.gather(learned, 2, order)
            vals = vals.masked_fill(~valid_b, 0.0)
            return idx_b, vals, valid_b, learned_b, vals

        idx_b = idx.unsqueeze(0).expand(B, -1, -1)
        fused = fused.masked_fill(~valid, 0.0)
        learned = learned.masked_fill(~valid, 0.0)
        return idx_b, fused, valid, learned, fused


class SparseMultiHeadGAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, use_graph_bias=True):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads.")

        self.heads = int(heads)
        self.dh = int(out_dim // heads)
        self.use_graph_bias = bool(use_graph_bias)

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(self.heads, self.dh))
        self.att_dst = nn.Parameter(torch.empty(self.heads, self.dh))
        self.graph_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, neighbor_idx, graph_prior, neighbor_valid):
        B, N, _ = x.shape
        K = int(neighbor_idx.shape[-1])
        H, Dh = self.heads, self.dh
        q = self.proj(x).view(B, N, H, Dh)

        # Additive GAT logit, evaluated per node before the neighbour gather.
        src = (q * self.att_src[None, None]).sum(-1)                 # [B,N,H]
        dst = (q * self.att_dst[None, None]).sum(-1)                 # [B,N,H]
        flat = neighbor_idx.reshape(B, N * K)
        dst_nbr = torch.gather(
            dst, 1, flat[..., None].expand(B, N * K, H)
        ).view(B, N, K, H)
        score = F.leaky_relu(src[:, :, None, :] + dst_nbr, 0.2)

        if self.use_graph_bias:
            score = score + self.graph_bias_scale * torch.log(
                graph_prior.clamp_min(1e-6)
            ).unsqueeze(-1)

        # Invalid boundary candidates must not participate in softmax.
        score = score.masked_fill(
            ~neighbor_valid.unsqueeze(-1), torch.finfo(score.dtype).min
        )
        attn = torch.softmax(score, dim=2)
        attn = self.dropout(attn)

        nbr = torch.gather(
            q.reshape(B, N, H * Dh), 1, flat[..., None].expand(B, N * K, H * Dh)
        ).view(B, N, K, H, Dh)
        out = (attn.unsqueeze(-1) * nbr).sum(dim=2).reshape(B, N, H * Dh)

        return F.gelu(self.norm(out + self.residual(x)))
