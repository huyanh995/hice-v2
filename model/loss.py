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

        # Multi-class path — targets is (B,) long or (B, C) float distribution
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
            # Multi-class path — softmax-based focal loss
            if targets.dtype == torch.long:
                y = F.one_hot(targets, n_cls).float()  # (B, C) one-hot
            else:
                y = targets                             # (B, C) soft distribution

            log_p = F.log_softmax(inputs, dim=1)
            p = log_p.exp()
            p_t = (y * p).sum(dim=1)                   # expected prob of true class(es)
            ce = -(y * log_p).sum(dim=1)               # soft CE over all classes
            mod = (1.0 - p_t).clamp_min(self.eps).pow(self.gamma)

            if self.alpha >= 0:
                # alpha weights fg mass vs bg mass
                alpha_t = self.alpha * (1 - y[:, 0]) + (1 - self.alpha) * y[:, 0]
                loss = alpha_t * mod * ce
            else:
                loss = mod * ce

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


class CTLoss(nn.Module):
    """
    Contact-State Transition loss.

    Operates on per-frame sigmoid outputs (r_t, p_t, c_t) produced by CTHead,
    and the 6 CT label arrays from ActionSpotDatasetCT.

    Components:
        L_trans  - Focal BCE on transition response r_t (masked by m_r)
        L_peak   - peak-contrast hinge: neighbours of a peak must be weaker by margin
        L_bg     - background sparsity: push non-event r_t towards 0
        L_pol    - BCE on polarity p_t (masked by m_p)
        L_rank   - contactness ranking: c_t should be higher after touch, lower after untouch
        L_contact- masked BCE on contactness c_t (optional; set lambda_c=0 to disable)
    """

    def __init__(
        self,
        lambda_peak: float = 1.0,
        lambda_bg: float = 0.3,
        lambda_pol: float = 1.0,
        lambda_rank: float = 0.5,
        lambda_contact: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        peak_margin: float = 0.2,
        rank_margin: float = 0.3,
        rank_delta: int = 4,
    ):
        super().__init__()
        self.lambda_peak = lambda_peak
        self.lambda_bg = lambda_bg
        self.lambda_pol = lambda_pol
        self.lambda_rank = lambda_rank
        self.lambda_contact = lambda_contact
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.peak_margin = peak_margin
        self.rank_margin = rank_margin
        self.rank_delta = rank_delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _focal_bce(self, logit: torch.Tensor, prob: torch.Tensor,
                   y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Focal BCE on logit (AMP-safe), masked by m. prob = sigmoid(logit)."""
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
        p_t = y * prob + (1 - y) * (1 - prob)
        mod = (1.0 - p_t.clamp_min(1e-8)).pow(self.gamma)
        alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
        loss = alpha_t * mod * bce * m
        denom = m.sum().clamp_min(1)
        return loss.sum() / denom

    def _peak_contrast(self, r: torch.Tensor, y_r: torch.Tensor, m_r: torch.Tensor) -> torch.Tensor:
        """
        For each frame t* where y_r==1.0 (peak center) and m_r==1,
        penalise neighbours Δ∈{±1..±4} that are within `peak_margin` of r[t*].

        L_peak = mean_{t*} mean_{Δ} max(0, margin - r[t*] + r[t*+Δ])
        """
        peak_mask = (y_r == 1.0) & (m_r == 1.0)
        B, L = r.shape
        loss = r.new_zeros(1)
        n = 0

        for delta in range(-4, 5):
            if delta == 0:
                continue
            if delta > 0:
                r_shift = torch.cat(
                    [r[:, delta:], r.new_zeros(B, delta)], dim=1)
                valid = peak_mask.clone()
                valid[:, L - delta:] = False
            else:
                d = -delta
                r_shift = torch.cat(
                    [r.new_zeros(B, d), r[:, :L - d]], dim=1)
                valid = peak_mask.clone()
                valid[:, :d] = False

            hinge = torch.clamp(
                self.peak_margin - r[valid] + r_shift[valid], min=0.0)
            loss = loss + hinge.sum()
            n += int(valid.sum().item())

        return loss / max(n, 1)

    def _ranking_loss(
        self,
        c: torch.Tensor,
        y_r: torch.Tensor,
        y_p: torch.Tensor,
        m_r: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each transition peak t*:
          - touch  (y_p=1): c_after  should exceed c_before  by rank_margin
          - untouch(y_p=0): c_before should exceed c_after   by rank_margin
        Windows: [t*-rank_delta, t*) and (t*, t*+rank_delta]
        """
        peak_mask = (y_r == 1.0) & (m_r == 1.0)
        B, L = c.shape
        delta = self.rank_delta
        loss = c.new_zeros(1)
        n = 0

        for b in range(B):
            idxs = peak_mask[b].nonzero(as_tuple=True)[0]
            for t in idxs.tolist():
                pre_s = max(0, t - delta)
                post_e = min(L, t + delta + 1)

                if t <= pre_s or t + 1 >= post_e:
                    continue

                c_before = c[b, pre_s:t].mean()
                c_after = c[b, t + 1:post_e].mean()
                polarity = y_p[b, t].item()

                if polarity == 1.0:  # touch: expect higher contact after
                    loss = loss + torch.clamp(
                        self.rank_margin - c_after + c_before, min=0.0)
                else:               # untouch: expect lower contact after
                    loss = loss + torch.clamp(
                        self.rank_margin - c_before + c_after, min=0.0)
                n += 1

        return loss / max(n, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ct_feat: torch.Tensor,  # (B, L, 3): [r_t, p_t, c_t] raw logits from CTHead
        y_r: torch.Tensor,      # (B, L)
        m_r: torch.Tensor,      # (B, L)
        y_p: torch.Tensor,      # (B, L)
        m_p: torch.Tensor,      # (B, L)
        y_c: torch.Tensor,      # (B, L)
        m_c: torch.Tensor,      # (B, L)
    ):
        r_logit = ct_feat[:, :, 0]  # (B, L) raw logits
        p_logit = ct_feat[:, :, 1]
        c_logit = ct_feat[:, :, 2]

        # Sigmoid probabilities needed for prob-based ops (peak, bg, ranking).
        # Computed in float32 to stay numerically stable; detach not needed.
        r = torch.sigmoid(r_logit.float())
        p = torch.sigmoid(p_logit.float())
        c = torch.sigmoid(c_logit.float())

        L_trans = self._focal_bce(r_logit.float(), r, y_r, m_r)
        L_peak = self._peak_contrast(r, y_r, m_r)

        bg_mask = (y_r == 0.0) & (m_r == 1.0)
        bg_sum = bg_mask.sum().clamp_min(1)
        L_bg = (r * bg_mask.float()).sum() / bg_sum

        pol_denom = m_p.sum().clamp_min(1)
        L_pol = (F.binary_cross_entropy_with_logits(
            p_logit.float(), y_p, reduction='none') * m_p).sum() / pol_denom

        L_rank = self._ranking_loss(c, y_r, y_p, m_r)

        con_denom = m_c.sum().clamp_min(1)
        L_contact = (F.binary_cross_entropy_with_logits(
            c_logit.float(), y_c, reduction='none') * m_c).sum() / con_denom

        total = (
            L_trans
            + self.lambda_peak * L_peak
            + self.lambda_bg * L_bg
            + self.lambda_pol * L_pol
            + self.lambda_rank * L_rank
            + self.lambda_contact * L_contact
        )

        detail = {
            'trans': L_trans.item(),
            'peak': L_peak.item(),
            'bg': L_bg.item(),
            'pol': L_pol.item(),
            'rank': L_rank.item(),
            'contact': L_contact.item(),
        }
        return total, detail
