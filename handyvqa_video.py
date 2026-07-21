#!/usr/bin/env python3
"""
Stitch one HanDyVQA video's final_scores_e2e/train_val/<video>.json + .npz
(real e2e_checkpoint_best.pt inference over Square_Frames_Version -- see
handyvqa_infer.py) into an mp4: the 224x224 model-input frame with the
object-of-interest heatmap overlaid on top, three stacked score graphs
(raw / NMS / SNMS) below, each with a moving playhead.

Usage:
    python handyvqa_video.py <video> [--out FILE] [--fps FPS]
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

FINAL_SCORES_ROOT = '/data/HanDyVQA/handyvqa_hice_infer/final_scores_e2e/train_val'
FRAMES_ROOT = '/data/HanDyVQA/Square_Frames_Version/Frames'

PANEL = 336            # each frame panel is square (224x224 model input), upscaled for legibility
GAP = 12                # gap between the two side-by-side frame panels
GRID = 7                # obj_heatmap grid
OVERLAY_ALPHA = 0.45
GRAPH_H = 220
HEADER_H = 40
DPI = 100
DEFAULT_FPS = 10


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('video', type=str, help='e.g. action_0001_frames')
    p.add_argument('--out', type=str, default=None,
                    help='Output .mp4 path. Defaults to viz_out/handyvqa_videos/<video>.mp4')
    p.add_argument('--fps', type=float, default=DEFAULT_FPS)
    return p.parse_args()


def events_to_frame_array(events, num_frames, score_columns):
    """Scatter a sparse (frame, label, score) event list back onto a dense
    (num_frames, num_classes) array -- zero everywhere except surviving events."""
    name_to_idx = {name: i for i, name in enumerate(score_columns)}
    arr = np.zeros((num_frames, len(score_columns)), dtype=np.float32)
    for e in events:
        arr[e['frame'], name_to_idx[e['label']]] = e['score']
    return arr


def load_frames(video, num_frames):
    frame_dir = os.path.join(FRAMES_ROOT, video)
    frames = []
    for i in range(num_frames):
        path = os.path.join(frame_dir, f'{i:06d}.jpg')
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            sys.exit(f'[ERROR] Missing frame file: {path}')
        frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
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

    with open(os.path.join(FINAL_SCORES_ROOT, f'{cli.video}.json')) as f:
        data = json.load(f)
    npz = np.load(os.path.join(FINAL_SCORES_ROOT, f'{cli.video}.npz'))
    obj_heatmap = npz['obj_heatmap']  # (num_frames, 7, 7)

    num_frames = data['num_frames']
    score_columns = data['score_columns']
    raw_scores = np.array(data['raw_scores'], dtype=np.float32)
    nms_scores = events_to_frame_array(data['nms_events'], num_frames, score_columns)
    snms_scores = events_to_frame_array(data['snms_events'], num_frames, score_columns)

    frames = load_frames(cli.video, num_frames)
    frame_idxs = np.arange(num_frames)

    vmin, vmax = float(obj_heatmap.min()), float(obj_heatmap.max())
    cell = PANEL // GRID

    width = PANEL * 2 + GAP
    height = HEADER_H + PANEL + GRAPH_H * 3

    graphs = []
    for title, arr in [('Raw (frame-averaged) scores', raw_scores),
                        ('NMS scores (window=1, threshold=0.01)', nms_scores),
                        ('Soft-NMS scores (window=3, threshold=0.01)', snms_scores)]:
        graph_rgb, frame_to_px = render_graph_base(frame_idxs, arr, score_columns, title, width, GRAPH_H)
        if graph_rgb.shape[1] != width:
            graph_rgb = cv2.resize(graph_rgb, (width, graph_rgb.shape[0]))
        graphs.append((cv2.cvtColor(graph_rgb, cv2.COLOR_RGB2BGR), frame_to_px))

    out_path = cli.out or os.path.join('viz_out', 'handyvqa_videos', f'{cli.video}.mp4')
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
        score_txt = '  '.join(f'{score_columns[c]}={raw_scores[i, c]:.2f}' for c in range(1, len(score_columns)))
        cv2.putText(header, f'frame {i}   raw: {score_txt}', (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        canvas = np.concatenate([header, top_row] + graph_frames, axis=0)
        ffmpeg.stdin.write(canvas.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f'[INFO] Wrote {num_frames} frames at {cli.fps} fps to {out_path}')


if __name__ == '__main__':
    main()
