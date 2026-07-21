#!/usr/bin/env python3
"""
Like handyvqa_video.py, but compares OLD (OASIS-precomputed, final_scores/)
vs NEW (real e2e_checkpoint_best.pt inference, final_scores_e2e/) raw scores
side by side instead of showing NMS/soft-NMS. Frame + obj_heatmap overlay
panels still come from the NEW model's own inference (OASIS has no spatial
heatmap to show).

Usage:
    python handyvqa_video_compare.py <video> [--split test|train_val] [--out FILE] [--fps FPS]
"""
import argparse
import os
import subprocess
import sys
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OLD_SCORES_ROOT = '/data/HanDyVQA/handyvqa_hice_infer/final_scores'
NEW_SCORES_ROOT = '/data/HanDyVQA/handyvqa_hice_infer/final_scores_e2e'
FRAMES_ROOT = '/data/HanDyVQA/Frames'

PANEL = 336             # each frame panel is square (224x224 model input), upscaled for legibility
GAP = 12                 # gap between the two side-by-side frame panels
GRID = 7                 # obj_heatmap grid
OVERLAY_ALPHA = 0.45
GRAPH_H = 220
HEADER_H = 40
DPI = 100
DEFAULT_FPS = 10


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('video', type=str, help='e.g. action_0001_frames')
    p.add_argument('--split', type=str, default=None, choices=['test', 'train_val'],
                    help='Defaults to whichever split has this video.')
    p.add_argument('--out', type=str, default=None,
                    help='Output .mp4 path. Defaults to viz_out/handyvqa_videos/<video>_compare.mp4')
    p.add_argument('--fps', type=float, default=DEFAULT_FPS)
    return p.parse_args()


def find_split(video, split):
    if split is not None:
        return split
    for s in ('test', 'train_val'):
        if os.path.exists(os.path.join(NEW_SCORES_ROOT, s, f'{video}.json')):
            return s
    sys.exit(f'[ERROR] No final_scores_e2e entry for "{video}" in test/ or train_val/.')


def crop_resize_frame(img_rgb, out_size=224):
    """Plain direct center crop, matching torchvision T.CenterCrop(out_size)'s
    offset convention -- what dataset/frame.py's eval path actually does."""
    H, W = img_rgb.shape[:2]
    crop_top = int(round((H - out_size) / 2.0))
    crop_left = int(round((W - out_size) / 2.0))
    return img_rgb[crop_top:crop_top + out_size, crop_left:crop_left + out_size]


def load_frames(video, num_frames):
    frame_dir = os.path.join(FRAMES_ROOT, video)
    frames = []
    for i in range(num_frames):
        path = os.path.join(frame_dir, f'{i:06d}.jpg')
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            sys.exit(f'[ERROR] Missing frame file: {path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(crop_resize_frame(img_rgb))
    return frames


def colorize(heat_grid, vmin, vmax, cell):
    norm = np.clip((heat_grid - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    rgba = matplotlib.colormaps['jet'](norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)


def make_overlay(frame_rgb, heat_grid, vmin, vmax, cell):
    heat_rgb = colorize(heat_grid, vmin, vmax, cell)
    if heat_rgb.shape[:2] != frame_rgb.shape[:2]:
        heat_rgb = cv2.resize(heat_rgb, (frame_rgb.shape[1], frame_rgb.shape[0]))
    return (frame_rgb.astype(np.float32) * (1 - OVERLAY_ALPHA)
            + heat_rgb.astype(np.float32) * OVERLAY_ALPHA).astype(np.uint8)


def render_graph_base(frame_idxs, scores_arr, score_columns, title, width_px, height_px):
    fg_classes = list(range(1, len(score_columns)))  # skip background
    colors = plt.cm.tab10.colors
    label_color = {score_columns[c]: colors[(c - 1) % len(colors)] for c in fg_classes}

    start, end = int(frame_idxs[0]), int(frame_idxs[-1]) + 1
    fig, ax = plt.subplots(figsize=(width_px / DPI, height_px / DPI), dpi=DPI)

    for c in fg_classes:
        name = score_columns[c]
        ax.plot(frame_idxs, scores_arr[:, c], label=name, linewidth=1.1, color=label_color[name])

    ax.axhline(0.5, color='lightgray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    ax.set_ylabel('Score', fontsize=8)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1.0)
    ax.set_xlim(start - 0.5, end - 0.5)
    ax.tick_params(axis='both', labelsize=7)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    graph_rgb = buf[:, :, :3].copy()

    x0_data, x1_data = start - 0.5, end - 0.5
    x0_px = ax.transData.transform((x0_data, 0))[0]
    x1_px = ax.transData.transform((x1_data, 0))[0]
    plt.close(fig)

    def frame_to_px(frame):
        t = (frame - x0_data) / (x1_data - x0_data)
        return int(round(x0_px + t * (x1_px - x0_px)))

    return graph_rgb, frame_to_px


def main():
    cli = get_args()
    split = find_split(cli.video, cli.split)

    with open(os.path.join(NEW_SCORES_ROOT, split, f'{cli.video}.json')) as f:
        new_data = json.load(f)
    with open(os.path.join(OLD_SCORES_ROOT, split, f'{cli.video}.json')) as f:
        old_data = json.load(f)
    npz = np.load(os.path.join(NEW_SCORES_ROOT, split, f'{cli.video}.npz'))
    obj_heatmap = npz['obj_heatmap']  # (num_frames, 7, 7)

    num_frames = new_data['num_frames']
    score_columns = new_data['score_columns']
    new_scores = np.array(new_data['raw_scores'], dtype=np.float32)
    old_scores = np.array(old_data['raw_scores'], dtype=np.float32)

    frames = load_frames(cli.video, num_frames)
    frame_idxs = np.arange(num_frames)

    vmin, vmax = float(obj_heatmap.min()), float(obj_heatmap.max())
    cell = PANEL // GRID

    width = PANEL * 2 + GAP
    height = HEADER_H + PANEL + GRAPH_H * 2

    graphs = []
    for title, arr in [('OLD (OASIS-precomputed) raw scores', old_scores),
                        ('NEW (e2e_checkpoint_best.pt) raw scores', new_scores)]:
        graph_rgb, frame_to_px = render_graph_base(frame_idxs, arr, score_columns, title, width, GRAPH_H)
        if graph_rgb.shape[1] != width:
            graph_rgb = cv2.resize(graph_rgb, (width, graph_rgb.shape[0]))
        graphs.append((cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2BGR), frame_to_px))

    out_path = cli.out or os.path.join('viz_out', 'handyvqa_videos', f'{cli.video}_compare.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
         '-r', str(cli.fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
         out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def label(img_bgr, text):
        cv2.putText(img_bgr, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return img_bgr

    for i in range(num_frames):
        frame_rgb = cv2.resize(frames[i], (PANEL, PANEL))
        overlay_rgb = make_overlay(frame_rgb, obj_heatmap[i], vmin, vmax, cell)

        raw_bgr = label(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy(), 'original')
        overlay_bgr = label(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), 'obj_heatmap overlay')
        gap_col = np.full((PANEL, GAP, 3), 255, dtype=np.uint8)
        top_row = np.concatenate([raw_bgr, gap_col, overlay_bgr], axis=1)

        graph_frames = []
        for graph_bgr, frame_to_px in graphs:
            g = graph_bgr.copy()
            px = frame_to_px(i)
            cv2.line(g, (px, 0), (px, GRAPH_H - 1), (0, 0, 0), 2)
            graph_frames.append(g)

        header = np.full((HEADER_H, width, 3), 30, dtype=np.uint8)
        score_txt = '  '.join(
            f'{score_columns[c]}: old={old_scores[i, c]:.2f} new={new_scores[i, c]:.2f}'
            for c in range(1, len(score_columns)))
        cv2.putText(header, f'frame {i}   {score_txt}', (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        canvas = np.concatenate([header, top_row] + graph_frames, axis=0)
        ffmpeg.stdin.write(canvas.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f'[INFO] Wrote {num_frames} frames at {cli.fps} fps to {out_path}')


if __name__ == '__main__':
    main()
