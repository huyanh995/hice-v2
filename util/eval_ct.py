"""
CT-mode evaluation: local-max decoding on transition response r_t.

Return signature matches evaluate() so main.py works unchanged:
  val pass  → float (avg mAP)
  test pass → (mAPs_list, tolerances)
"""

import os
import pickle
from collections import defaultdict

import numpy as np
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.eval import (
    INFERENCE_BATCH_SIZE,
    TOLERANCES,
    WINDOWS,
    non_maximum_supression,
    soft_non_maximum_supression,
)
from util.io import store_json
from util.score import compute_mAPs


# ──────────────────────────────────────────────────────────────────────────────
# Local-max decoding
# ──────────────────────────────────────────────────────────────────────────────

def _local_max_events(r, p, classes_inv, threshold=0.05):
    """
    Find local maxima of r and emit one touch + one untouch candidate per peak.
    score_touch = r * p,  score_untouch = r * (1 - p).
    """
    L = len(r)
    events = []
    for t in range(L):
        left_ok  = (t == 0)      or (r[t] > r[t - 1])
        right_ok = (t == L - 1)  or (r[t] >= r[t + 1])
        if left_ok and right_ok and r[t] >= threshold:
            events.append({'label': classes_inv[1], 'frame': t,
                           'score': float(r[t] * p[t])})
            events.append({'label': classes_inv[2], 'frame': t,
                           'score': float(r[t] * (1.0 - p[t]))})
    return events


# ──────────────────────────────────────────────────────────────────────────────
# Frame-level F1 helpers (approximate; uses best-score peak per frame)
# ──────────────────────────────────────────────────────────────────────────────

class _F1Counter:
    def __init__(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def update(self, true_label, pred_label):
        if pred_label != 0:
            if true_label != 0:
                self.tp[None] += 1
            else:
                self.fp[None] += 1
            if pred_label == true_label:
                self.tp[pred_label] += 1
            else:
                self.fp[pred_label] += 1
                if true_label != 0:
                    self.fn[true_label] += 1
        elif true_label != 0:
            self.fn[None] += 1
            self.fn[true_label] += 1

    def f1(self, k):
        denom = self.tp[k] + 0.5 * self.fp[k] + 0.5 * self.fn[k]
        return self.tp[k] / max(denom, 1)

    def tp_fp_fn(self, k):
        return self.tp[k], self.fp[k], self.fn[k]


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
    classes_inv = {v: k for k, v in classes.items()}

    # Per-video accumulators
    r_accum   = {}
    p_accum   = {}
    support   = {}
    for video, video_len, _ in dataset.videos:
        r_accum[video]  = np.zeros(video_len, np.float32)
        p_accum[video]  = np.zeros(video_len, np.float32)
        support[video]  = np.zeros(video_len, np.int32)

    dataloader = DataLoader(
        dataset, num_workers=8, pin_memory=True, batch_size=INFERENCE_BATCH_SIZE
    )

    print(f'[CT Eval] Running inference (batch={INFERENCE_BATCH_SIZE}) …')
    for clip in tqdm(dataloader):
        _, _, raw = model.predict(
            clip['frame'],
            clip['left_patches'], clip['right_patches'],
            clip['left_grasp'],   clip['right_grasp'],
        )

        ct = raw['ct_feat']   # numpy (B, L, 3) — already sigmoid probs
        B  = ct.shape[0]

        for i in range(B):
            video = clip['video'][i]
            start = int(clip['start'][i].item())
            r_clip = ct[i, :, 0]
            p_clip = ct[i, :, 1]

            if start < 0:
                r_clip = r_clip[-start:]
                p_clip = p_clip[-start:]
                start  = 0

            vid_len = r_accum[video].shape[0]
            end     = min(start + len(r_clip), vid_len)
            r_clip  = r_clip[:end - start]
            p_clip  = p_clip[:end - start]

            r_accum[video][start:end] += r_clip
            p_accum[video][start:end] += p_clip
            support[video][start:end] += 1

    # Normalise and decode per video
    pred_events_all = []
    for video, video_len, fps in dataset.videos:
        sup          = np.maximum(support[video], 1)
        r            = r_accum[video] / sup
        p            = p_accum[video] / sup
        local_maxima = _local_max_events(r, p, classes_inv)
        pred_events_all.append({'video': video, 'events': local_maxima, 'fps': fps})

    # ── Validation pass ───────────────────────────────────────────────────────
    if not test:
        pred_nms = non_maximum_supression(
            pred_events_all, window=windows[0], threshold=0.05)
        mAPs, _, _, _ = compute_mAPs(
            dataset.labels, pred_nms,
            tolerances=tolerances, printed=printed, plot_pr=False,
        )
        return float(np.mean(mAPs))

    # ── Test pass ─────────────────────────────────────────────────────────────

    # Frame-level F1 using r > 0.5 threshold
    f1_ctr    = _F1Counter()
    total_err = 0
    total_frm = 0
    gt_labels = {}
    for video, video_len, _ in dataset.videos:
        gt_labels[video] = dataset.get_labels(video)   # (video_len,) int

    for video, video_len, fps in dataset.videos:
        sup = np.maximum(support[video], 1)
        r   = r_accum[video] / sup
        p   = p_accum[video] / sup
        gt  = gt_labels[video]

        # Per-frame prediction: argmax of [1-r, r*p, r*(1-p)]
        score_bg     = 1.0 - r
        score_touch  = r * p
        score_untouch = r * (1.0 - p)
        scores = np.stack([score_bg, score_touch, score_untouch], axis=1)
        pred   = np.argmax(scores, axis=1)

        total_err += int(np.sum(gt != pred))
        total_frm += video_len
        for t in range(video_len):
            f1_ctr.update(int(gt[t]), int(pred[t]))

    frame_err = total_err / max(total_frm, 1)

    msg = ''
    msg += '=== Frame-level results ===\n'
    msg += 'Error (frame-level): {:0.2f}\n'.format(frame_err * 100)

    rows = [['any', f1_ctr.f1(None) * 100, *f1_ctr.tp_fp_fn(None)]]
    for cls_name in sorted(classes):
        k = classes[cls_name]
        rows.append([cls_name, f1_ctr.f1(k) * 100, *f1_ctr.tp_fp_fn(k)])
    msg += tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                    floatfmt='0.2f') + '\n\n'

    # mAP without NMS
    msg += '=== CT Results on {} (w/o NMS) ===\n'.format(split)
    mAPs_raw, _, tab_raw, fig_raw = compute_mAPs(
        dataset.labels, pred_events_all,
        tolerances=tolerances, printed=printed, plot_pr=True,
    )
    msg += tab_raw + '\nAvg mAP (across tolerances): {:0.2f}\n\n'.format(
        np.mean(mAPs_raw) * 100)

    # mAP with NMS
    nms_pred = non_maximum_supression(
        pred_events_all, window=windows[0], threshold=0.05)
    msg += '=== CT Results on {} (w/ NMS{}) ===\n'.format(split, windows[0])
    mAPs_nms, _, tab_nms, fig_nms = compute_mAPs(
        dataset.labels, nms_pred,
        tolerances=tolerances, printed=printed, plot_pr=True,
    )
    msg += tab_nms + '\nAvg mAP (across tolerances): {:0.2f}\n\n'.format(
        np.mean(mAPs_nms) * 100)

    # mAP with Soft-NMS
    snms_pred = soft_non_maximum_supression(
        pred_events_all, window=windows[1], threshold=0.05)
    msg += '=== CT Results on {} (w/ SNMS{}) ===\n'.format(split, windows[1])
    mAPs_snms, _, tab_snms, fig_snms = compute_mAPs(
        dataset.labels, snms_pred,
        tolerances=tolerances, printed=printed, plot_pr=True,
    )
    msg += tab_snms + '\nAvg mAP (across tolerances): {:0.2f}\n\n'.format(
        np.mean(mAPs_snms) * 100)

    print(msg)

    if save_dir is not None:
        save_pred = os.path.join(save_dir, 'pred-{}'.format(split.lower()))
        store_json(save_pred + '.json',      pred_events_all)
        store_json(save_pred + '_nms.json',  nms_pred)
        store_json(save_pred + '_snms.json', snms_pred)

        with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
            f.write(msg)

        fig_dir = os.path.join(save_dir, 'figs')
        os.makedirs(fig_dir, exist_ok=True)
        fig_raw.savefig( os.path.join(fig_dir, f'{split}_PR_Curves.png'),      dpi=300, bbox_inches='tight')
        fig_nms.savefig( os.path.join(fig_dir, f'{split}_PR_Curves_NMS.png'),  dpi=300, bbox_inches='tight')
        fig_snms.savefig(os.path.join(fig_dir, f'{split}_PR_Curves_SNMS.png'), dpi=300, bbox_inches='tight')

    return mAPs_nms, tolerances   # matches evaluate() signature for main.py
