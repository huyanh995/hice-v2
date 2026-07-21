#!/usr/bin/env python3
"""
Real model inference (not the external OASIS pkl dumps) over HanDyVQA's
original Frames directory, using the verified e2e_checkpoint_best.pt
(TouchMoment-v2, touch+untouch, obj_head=True, crop_dim=224). crop_dim=224
against native-resolution frames means dataset/frame.py's CenterCrop does a
real crop here -- same standard protocol used for any other eval on this repo.

hand_anno.json (native-resolution box coords, correct against these native
frames) only covers the train_val split (1661 videos). The test split (9433
videos) has no hand annotations at all -- those videos get synthetic
"no hand visible" placeholders per frame, so the hand cross-attention is
padding-masked out entirely for them (hand-blind fallback, agreed on earlier).

Use --shard/--num_shards to split the full ~11092-video workload across
multiple GPUs, e.g.:
    python handyvqa_infer.py --device cuda:0 --shard 0 --num_shards 2
    python handyvqa_infer.py --device cuda:1 --shard 1 --num_shards 2

Per video, saves into final_scores_e2e/<split>/ (split = test or train_val,
matching the original OASIS handyvqa_hice_infer/<split>/ membership):
  <video>.json  -- raw_scores, raw_pred, raw_events, nms_events, snms_events
      (same schema/constants as score_handyvqa.py)
  <video>.npz   -- obj_heatmap (num_frames, 7, 7), the model's own
      object-of-interest branch output (the "attention map").
"""
import argparse
import copy
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.frame import ActionSpotVideoDataset
from model.model import TDEEDModel
from util.dataset import load_classes
from util.io import store_json

FRAME_DIR = '/data/HanDyVQA/Frames'
HAND_ANNO_PATH = '/data/HanDyVQA/hand_anno.train_val.json'  # only train_val is covered
OASIS_SPLIT_ROOT = '/data/HanDyVQA/handyvqa_hice_infer'      # test/ vs train_val/ membership
CHECKPOINT = '/data/hice-v2/checkpoints/e2e_checkpoint_best.pt'
OUT_ROOT = '/data/HanDyVQA/handyvqa_hice_infer/final_scores_e2e'
CLASSES_PATH = 'data/TouchMoment-v2/class.txt'

CLIP_LEN = 40
OVERLAP_LEN = CLIP_LEN // 4 * 3  # 30 -- matches main.py's real test-time construction
STRIDE = 1
CROP_DIM = 224
BATCH_SIZE = 8

HIGH_RECALL_THRESHOLD = 0.01
NMS_WINDOW = 1
SNMS_WINDOW = 3
NMS_THRESHOLD = 0.01
SNMS_THRESHOLD = 0.01


def build_args(device):
    return argparse.Namespace(
        modality='rgb', temporal_arch='ed_sgp_mixer', radi_displacement=4,
        feature_arch='rny008_gsf', clip_len=CLIP_LEN, temporal_shift=True, share_enc=False,
        n_layers=3, sgp_ks=9, sgp_r=4, num_classes=2, obj_head=True, obj_stage1=False,
        obj_loss_weight=0.15, obj_fg_weight=10.0, lambda_presence=0.75, presence_tau=0.1,
        grasp_loss=True, bi_interp_post=True, crop_dim=CROP_DIM, fg_weight=5.0,
        focal_alpha=0.9, focal_gamma=2.0, loss_type='focal', mixup=False, pretrain=None,
        compile=False, amp=True, device=device,
    )


def non_maximum_supression(events, window, threshold=0.0):
    v = copy.deepcopy(events)
    out = []
    while len(v) > 0:
        e1 = max(v, key=lambda x: x['score'])
        if e1['score'] < threshold:
            break
        pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
        out.append(copy.deepcopy(e1))
        v.pop(pos1)
        list_pos = [pos for pos, e in enumerate(v)
                    if (e['frame'] >= e1['frame'] - window) and (e['frame'] <= e1['frame'] + window)]
        for pos in list_pos[::-1]:
            v.pop(pos)
    out.sort(key=lambda x: x['frame'])
    return out


def soft_non_maximum_supression(events, window, threshold=0.01):
    v = copy.deepcopy(events)
    out = []
    while len(v) > 0:
        e1 = max(v, key=lambda x: x['score'])
        if e1['score'] < threshold:
            break
        pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
        out.append(copy.deepcopy(e1))
        list_pos = [pos for pos, e in enumerate(v)
                    if (e['frame'] >= e1['frame'] - window) and (e['frame'] <= e1['frame'] + window)]
        for pos in list_pos:
            v[pos]['score'] = v[pos]['score'] * (np.abs(e1['frame'] - v[pos]['frame'])) ** 2 / (window ** 2)
        v.pop(pos1)
    out.sort(key=lambda x: x['frame'])
    return out


def build_video_split_map():
    """video -> 'test' | 'train_val', from the original OASIS split membership."""
    mapping = {}
    for split in ('test', 'train_val'):
        split_dir = os.path.join(OASIS_SPLIT_ROOT, split)
        for fname in os.listdir(split_dir):
            if fname.endswith('.pkl'):
                mapping[fname[:-len('.pkl')]] = split
    return mapping


def build_manifest(videos):
    entries = []
    for v in videos:
        n = len([f for f in os.listdir(os.path.join(FRAME_DIR, v)) if f.endswith('.jpg')])
        entries.append({'video': v, 'num_frames': n, 'fps': 1, 'events': []})
    return entries


def build_hand_anno(videos, video_lens):
    """Real annotations for train_val videos; synthetic 'no hand visible' per-frame
    placeholders for test videos (no annotations exist for them at any resolution)."""
    with open(HAND_ANNO_PATH) as f:
        real = json.load(f)

    merged = {}
    n_real, n_placeholder = 0, 0
    for v in videos:
        if v in real:
            merged[v] = real[v]
            n_real += 1
        else:
            n = video_lens[v]
            merged[v] = {f'{i:06d}.jpg': {'left hand': None, 'right hand': None} for i in range(n)}
            n_placeholder += 1
    print(f'[INFO] hand_anno: {n_real} videos with real annotations, '
          f'{n_placeholder} videos hand-blind (placeholder).')
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N videos (for testing).')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    cli = parser.parse_args()

    video_split = build_video_split_map()
    videos = sorted(video_split.keys())
    videos = videos[cli.shard::cli.num_shards]
    if cli.limit:
        videos = videos[:cli.limit]
    print(f'[INFO] Shard {cli.shard}/{cli.num_shards}: {len(videos)} videos.')

    for split in ('test', 'train_val'):
        os.makedirs(os.path.join(OUT_ROOT, split), exist_ok=True)

    manifest = build_manifest(videos)
    manifest_path = f'/tmp/handyvqa_manifest_shard{cli.shard}.json'
    store_json(manifest_path, manifest)
    video_lens = {e['video']: e['num_frames'] for e in manifest}

    hand_anno = build_hand_anno(videos, video_lens)
    hand_anno_path = f'/tmp/handyvqa_hand_anno_shard{cli.shard}.json'
    store_json(hand_anno_path, hand_anno)

    classes = load_classes(CLASSES_PATH)
    score_columns = ['background'] + sorted(classes, key=classes.get)
    print('[INFO] score_columns:', score_columns)

    dataset = ActionSpotVideoDataset(
        classes, manifest_path, FRAME_DIR, modality='rgb', clip_len=CLIP_LEN,
        overlap_len=OVERLAP_LEN, stride=STRIDE, crop_dim=CROP_DIM, dataset='HanDyVQA',
        hand_anno_path=hand_anno_path)

    args = build_args(cli.device)
    model = TDEEDModel(args=args)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model._model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model._model.eval()
    print(f'[INFO] Loaded checkpoint (epoch {ckpt["epoch"]}) -- strict load OK.')

    scores_acc = {v: (np.zeros((n, len(score_columns)), dtype=np.float64), np.zeros(n, dtype=np.int32))
                  for v, n in video_lens.items()}
    obj_acc = {v: (np.zeros((n, 7, 7), dtype=np.float64), np.zeros(n, dtype=np.int32))
               for v, n in video_lens.items()}

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    n_clips = 0
    with torch.no_grad():
        for clip in loader:
            frame = clip['frame'].to(cli.device).float()
            left_patches = clip['left_patches'].to(cli.device).float()
            right_patches = clip['right_patches'].to(cli.device).float()
            left_grasp = clip['left_grasp'].to(cli.device).float()
            right_grasp = clip['right_grasp'].to(cli.device).float()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                res, _ = model._model(frame, left_patches, right_patches, left_grasp, right_grasp, inference=True)

            im_feat = res['im_feat'].float()
            displ_feat = res['displ_feat'].float() if 'displ_feat' in res else None
            pred_scores = model.process_prediction(im_feat, displ_feat).cpu().numpy()  # (B, L, 3)
            obj_heatmap = torch.sigmoid(res['obj_heatmap']).float().cpu().numpy()       # (B, L, 7, 7)

            B = frame.shape[0]
            for i in range(B):
                video = clip['video'][i]
                start = clip['start'][i].item()
                n = video_lens[video]

                p_scores = pred_scores[i]
                p_obj = obj_heatmap[i]
                if start < 0:
                    p_scores = p_scores[-start:]
                    p_obj = p_obj[-start:]
                    start = 0
                end = start + p_scores.shape[0]
                if end >= n:
                    end = n
                    p_scores = p_scores[:end - start]
                    p_obj = p_obj[:end - start]

                s_arr, s_sup = scores_acc[video]
                s_arr[start:end] += p_scores
                s_sup[start:end] += 1

                o_arr, o_sup = obj_acc[video]
                o_arr[start:end] += p_obj
                o_sup[start:end] += 1

            n_clips += B
            if n_clips % 500 < BATCH_SIZE:
                print(f'[INFO] {n_clips} clips processed...', flush=True)

    print(f'[INFO] Done inference: {n_clips} clips total. Merging + writing per-video outputs...')

    for video in videos:
        s_arr, s_sup = scores_acc[video]
        s_sup_safe = s_sup.copy()
        s_sup_safe[s_sup_safe == 0] = 1
        scores = s_arr / s_sup_safe[:, None]
        raw_pred = np.argmax(scores, axis=1).tolist()

        o_arr, o_sup = obj_acc[video]
        o_sup_safe = o_sup.copy()
        o_sup_safe[o_sup_safe == 0] = 1
        obj_heatmap = (o_arr / o_sup_safe[:, None, None]).astype(np.float32)

        num_frames = video_lens[video]
        raw_events = []
        for fi in range(num_frames):
            for c in range(1, len(score_columns)):
                s = scores[fi, c]
                if s >= HIGH_RECALL_THRESHOLD:
                    raw_events.append({'label': score_columns[c], 'frame': fi, 'score': float(s)})

        events_by_label = defaultdict(list)
        for e in raw_events:
            events_by_label[e['label']].append(e)
        nms_events, snms_events = [], []
        for label, evs in events_by_label.items():
            nms_events.extend(non_maximum_supression(evs, window=NMS_WINDOW, threshold=NMS_THRESHOLD))
            snms_events.extend(soft_non_maximum_supression(evs, window=SNMS_WINDOW, threshold=SNMS_THRESHOLD))
        nms_events.sort(key=lambda x: x['frame'])
        snms_events.sort(key=lambda x: x['frame'])

        out = {
            'video': video, 'num_frames': num_frames, 'score_columns': score_columns,
            'raw_scores': scores.tolist(), 'raw_pred': raw_pred,
            'raw_events': raw_events, 'nms_events': nms_events, 'snms_events': snms_events,
        }
        split = video_split[video]
        with open(os.path.join(OUT_ROOT, split, f'{video}.json'), 'w') as f:
            json.dump(out, f)
        np.savez_compressed(os.path.join(OUT_ROOT, split, f'{video}.npz'), obj_heatmap=obj_heatmap)

    print(f'[DONE] Shard {cli.shard}/{cli.num_shards}: wrote {len(videos)} video results to {OUT_ROOT}/{{test,train_val}}')


if __name__ == '__main__':
    main()
