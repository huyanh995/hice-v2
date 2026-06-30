"""
CT-mode evaluation: local-max decoding on transition response r_t.

Usage:
    mAP = evaluate_ct(model, dataset, split, classes, tolerances, windows)
"""

import copy
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.eval import (
    INFERENCE_BATCH_SIZE,
    TOLERANCES,
    WINDOWS,
    non_maximum_supression,
    soft_non_maximum_supression,
)
from util.score import compute_mAPs


# ──────────────────────────────────────────────────────────────────────────────
# Local-max decoding
# ──────────────────────────────────────────────────────────────────────────────

def _local_max_decode(r, p, classes_inv, video, threshold=0.05):
    """
    Find local maxima of r_t and emit candidate events.

    A frame t is a local max if:
        r[t] > r[t-1]  AND  r[t] >= r[t+1]   (boundary frames: relaxed)

    score_touch   = r[t] * p[t]
    score_untouch = r[t] * (1 - p[t])

    Returns list of {'label': str, 'frame': int, 'score': float}.
    """
    L = r.shape[0]
    events = []
    for t in range(L):
        left_ok = (t == 0) or (r[t] > r[t - 1])
        right_ok = (t == L - 1) or (r[t] >= r[t + 1])
        if left_ok and right_ok and r[t] >= threshold:
            score_touch = float(r[t] * p[t])
            score_untouch = float(r[t] * (1.0 - p[t]))
            touch_cls = classes_inv.get(1, 'touch')
            untouch_cls = classes_inv.get(2, 'untouch')
            events.append({'label': touch_cls, 'frame': t, 'score': score_touch})
            events.append({'label': untouch_cls, 'frame': t, 'score': score_untouch})
    return events


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_ct(
    model,
    dataset,
    split,
    classes,
    tolerances=TOLERANCES,
    windows=WINDOWS,
    printed=True,
    test=False,
    save_dir=None,
):
    """
    Evaluate a CT model on the given video dataset.

    Accumulates (r_t, p_t) predictions per video using overlapping clips,
    then applies local-max decoding to produce candidate events.
    """
    classes_inv = {v: k for k, v in classes.items()}

    # Per-video accumulators: sum of (r, p) and overlap count
    r_accum = {}
    p_accum = {}
    support = {}
    for video, video_len, _ in dataset.videos:
        r_accum[video] = np.zeros(video_len, np.float32)
        p_accum[video] = np.zeros(video_len, np.float32)
        support[video] = np.zeros(video_len, np.int32)

    dataloader = DataLoader(
        dataset, num_workers=8, pin_memory=True, batch_size=INFERENCE_BATCH_SIZE
    )

    print(f'[CT Eval] Running inference (batch={INFERENCE_BATCH_SIZE}) …')
    for clip in tqdm(dataloader):
        left_patches = clip['left_patches']
        right_patches = clip['right_patches']
        left_grasp = clip['left_grasp']
        right_grasp = clip['right_grasp']

        _, _, raw = model.predict(
            clip['frame'], left_patches, right_patches, left_grasp, right_grasp
        )

        # raw['ct_feat']: (B, L, 3) numpy array — [r, p, c] after sigmoid
        ct = raw['ct_feat']  # numpy (B, L, 3)
        B = ct.shape[0]

        for i in range(B):
            video = clip['video'][i]
            start = int(clip['start'][i].item())
            r_clip = ct[i, :, 0]  # (L,)
            p_clip = ct[i, :, 1]  # (L,)

            # Trim padding
            if start < 0:
                r_clip = r_clip[-start:]
                p_clip = p_clip[-start:]
                start = 0

            vid_len = r_accum[video].shape[0]
            end = min(start + r_clip.shape[0], vid_len)
            r_clip = r_clip[:end - start]
            p_clip = p_clip[:end - start]

            r_accum[video][start:end] += r_clip
            p_accum[video][start:end] += p_clip
            support[video][start:end] += 1

    # Normalise by overlap count, then decode
    pred_events_all = []
    for video, video_len, fps in dataset.videos:
        sup = support[video].copy()
        sup[sup == 0] = 1  # avoid divide by zero for padding regions

        r = r_accum[video] / sup
        p = p_accum[video] / sup

        local_maxima = _local_max_decode(r, p, classes_inv, video)
        pred_events_all.append({
            'video': video,
            'events': local_maxima,
            'fps': fps,
        })

    # ── Validation pass: NMS + mAP ────────────────────────────────────────────
    if not test:
        pred_nms = non_maximum_supression(
            pred_events_all, window=windows[0], threshold=0.05)
        mAPs, _, _, _ = compute_mAPs(
            dataset.labels, pred_nms, tolerances=tolerances,
            printed=printed, plot_pr=False,
        )
        return float(np.mean(mAPs))

    # ── Test pass: report all variants ────────────────────────────────────────
    from tabulate import tabulate

    lines = [f'=== CT Results on {split} ===']

    lines.append('\n--- Raw local-max (no NMS) ---')
    mAPs_raw, _, tab_raw, _ = compute_mAPs(
        dataset.labels, pred_events_all, tolerances=tolerances,
        printed=printed, plot_pr=False,
    )
    lines.append(tab_raw)
    lines.append(f'Avg mAP: {np.mean(mAPs_raw)*100:.2f}')

    nms_pred = non_maximum_supression(
        pred_events_all, window=windows[0], threshold=0.05)
    lines.append(f'\n--- NMS (window={windows[0]}) ---')
    mAPs_nms, _, tab_nms, _ = compute_mAPs(
        dataset.labels, nms_pred, tolerances=tolerances,
        printed=printed, plot_pr=False,
    )
    lines.append(tab_nms)
    lines.append(f'Avg mAP: {np.mean(mAPs_nms)*100:.2f}')

    snms_pred = soft_non_maximum_supression(
        pred_events_all, window=windows[1], threshold=0.05)
    lines.append(f'\n--- Soft-NMS (window={windows[1]}) ---')
    mAPs_snms, _, tab_snms, _ = compute_mAPs(
        dataset.labels, snms_pred, tolerances=tolerances,
        printed=printed, plot_pr=False,
    )
    lines.append(tab_snms)
    lines.append(f'Avg mAP: {np.mean(mAPs_snms)*100:.2f}')

    msg = '\n'.join(lines)
    print(msg)

    if save_dir is not None:
        import os
        with open(os.path.join(save_dir, f'ct_eval_{split}.txt'), 'w') as f:
            f.write(msg)

    return float(np.mean(mAPs_nms)), tolerances
