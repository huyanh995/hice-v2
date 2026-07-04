import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    BCE Loss but supports soft targets

    """
    def __init__(self, fg_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.w_bg = 1.0
        self.w_fg = float(fg_weight)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs:  (B, C) logits; C=2 for binary, C>2 for multi-class
        targets: (B,) float in [0,1] (binary soft) or long (hard), or
                 (B, C) float distribution (multi-class soft)
        """
        n_cls = inputs.shape[1]
        logp = F.log_softmax(inputs, dim=1)           # (B, C)

        if n_cls == 2:
            # Binary path — targets is scalar float in [0,1] or long {0,1}
            y = targets.float() if targets.dtype == torch.long else targets
            loss = -((1 - y) * self.w_bg * logp[:, 0] +
                      y       * self.w_fg * logp[:, 1])
            if self.reduction == 'none':
                return loss
            if self.reduction == 'sum':
                return loss.sum()
            eff_w = (1 - y) * self.w_bg + y * self.w_fg
            return loss.sum() / eff_w.sum().clamp_min(1e-8)

        # Multi-class path — softmax-distribution based (targets must sum to 1 per
        # row). The gaussian soft labels used for margin-vs-background training
        # (see FocalLoss) are NOT distributions, so this path is unsafe for them.
        if n_cls > 2:
            raise NotImplementedError(
                "BCELoss multi-class path assumes softmax-distribution targets; "
                "use FocalLoss (loss_type='focal') for multi-label gaussian soft labels.")
        if targets.dtype == torch.long:
            y = F.one_hot(targets, n_cls).float()     # (B, C) one-hot
        else:
            y = targets                                # (B, C) soft distribution

        w = torch.full((n_cls,), self.w_fg, device=inputs.device, dtype=logp.dtype)
        w[0] = self.w_bg

        loss = -(y * w * logp).sum(dim=1)             # (B,)

        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return loss.sum()

        eff_w = (y * w).sum(dim=1)
        return loss.sum() / eff_w.sum().clamp_min(1e-8)

class FocalLoss(nn.Module):
    """
    Binary focal loss for a (B,2)=[bg, fg] logit head.
    Supports hard or soft targets y in [0,1].

    FL = alpha_t * (1 - p_t)^gamma * BCEWithLogits(m, y),
    where m = z_fg - z_bg, p = sigmoid(m),
          p_t = y*p + (1-y)*(1-p),
          alpha_t = alpha*y + (1-alpha)*(1-y)  (if alpha>=0).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        if not (-1.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1] or -1 to disable.")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none' | 'mean' | 'sum'")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs:  (B, C) logits; C=2 for binary, C>2 for multi-class
        targets: (B,) float in [0,1] (binary soft) or long (hard), or
                 (B, C) float distribution (multi-class soft)
        """
        n_cls = inputs.shape[1]

        if n_cls == 2:
            # Binary path — stable margin-based formulation
            y = targets.float()
            m = inputs[:, 1] - inputs[:, 0]
            bce = F.binary_cross_entropy_with_logits(m, y, reduction="none")
            p = torch.sigmoid(m)
            p_t = y * p + (1 - y) * (1 - p)
            mod = (1.0 - p_t).clamp_min(self.eps).pow(self.gamma)
            if self.alpha >= 0:
                alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
                loss = alpha_t * mod * bce
            else:
                loss = mod * bce
        else:
            # Multi-class path — multi-label margin-vs-background focal loss.
            # Each fg class c is scored independently by its margin against the
            # shared background logit, m_c = z_c - z_bg, and trained with the same
            # binary focal math as the n_cls==2 branch above (per-class proper
            # scoring rule; reduces exactly to the n_cls==2 branch at C=1).
            if targets.dtype == torch.long:
                y = torch.zeros(targets.shape[0], n_cls - 1, device=inputs.device, dtype=inputs.dtype)
                fg = targets > 0
                y[fg, targets[fg] - 1] = 1.0            # (B, C-1); targets==0 -> all-zero row
            else:
                y = targets[:, 1:]                       # (B, C-1) soft labels; column 0 ignored

            m = inputs[:, 1:] - inputs[:, :1]           # (B, C-1)
            bce = F.binary_cross_entropy_with_logits(m, y, reduction="none")
            p = torch.sigmoid(m)
            p_t = y * p + (1 - y) * (1 - p)
            mod = (1.0 - p_t).clamp_min(self.eps).pow(self.gamma)
            if self.alpha >= 0:
                alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
                loss = alpha_t * mod * bce
            else:
                loss = mod * bce
            loss = loss.mean(dim=1)                     # (B,) mean over classes

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class FocalLossPlain(nn.Module):
    """
    Focal Loss (plain two-class version using softmax probabilities)
    Works for binary classification with logits of shape (B, 2).

    FL = - α_t * (1 - p_t)^γ * [(1 - y) * log p_0 + y * log p_1]
    where p = softmax(logits), p_t = y*p_1 + (1 - y)*p_0

    Args:
        alpha (float): Class-balance weight (0–1). -1 disables α term.
        gamma (float): Focusing parameter.
        reduction (str): 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", eps=1e-8):
        super().__init__()
        if not (-1 <= alpha <= 1):
            raise ValueError("alpha must be in [0,1] or -1 to disable.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("Invalid reduction mode.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, 2) logits = [bg, fg]
            targets: (B,) float in [0,1] (soft or hard)
        """
        y = targets.float()
        log_p = F.log_softmax(inputs, dim=1)       # (B,2)
        p = log_p.exp()                            # (B,2)

        p0, p1 = p[:, 0], p[:, 1]
        log_p0, log_p1 = log_p[:, 0], log_p[:, 1]

        # Probability of correct class
        p_t = y * p1 + (1 - y) * p0                # (B,)
        # Base CE term
        ce = (1 - y) * log_p0 + y * log_p1         # (B,)
        # Modulating factor (1 - p_t)^γ
        mod = (1.0 - p_t).clamp_min(self.eps).pow(self.gamma)

        loss = -mod * ce                           # apply focal modulation

        if self.alpha >= 0:
            alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # (B,)
