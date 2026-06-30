"""
CT (Contact-State Transition) dataset.

Stores per-clip (carry_in_state, in_window_events) at cache time,
computes the 6 CT label arrays on-the-fly in _get_one().

Cache key: LEN{clip_len}CT_SPLIT{split}  (distinct from baseline cache).
"""

import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

from dataset.frame import (
    DEFAULT_PAD_LEN,
    ActionSpotDataset,
    ActionSpotVideoDataset,
)

# Transition response shoulder weights: {delta_from_peak: weight}
_SHOULDER = {-1: 0.25, 0: 1.0, 1: 0.25}


def compute_ct_labels(clip_len: int, carry_in: bool, events: list):
    """
    Build 6 CT label arrays.

    All events are treated as real transitions (double-touch = both hands; no masking).
    Contact state after each event is determined by the event label
    (touch → 1, untouch → 0), regardless of carry-in or prior state.

    Args:
        clip_len: number of frames in the clip.
        carry_in: contact state just before the clip window starts.
        events: list of {'label': 'touch'|'untouch', 'label_idx': int} within the clip.

    Returns (each np.float32 of shape (clip_len,)):
        y_r: transition response target  (peak=1.0, shoulder=0.25, bg=0)
        m_r: transition mask             (1=use, 0=ignore; always 1 here)
        y_p: polarity target             (1=touch, 0=untouch, 0=bg — only m_p frames matter)
        m_p: polarity mask               (1 only at transition frame)
        y_c: contactness target          (0/1 piecewise, set by event labels)
        m_c: contactness mask            (1=use, 0=ignore; always 1 here)
    """
    y_r = np.zeros(clip_len, np.float32)
    m_r = np.ones(clip_len, np.float32)
    y_p = np.zeros(clip_len, np.float32)
    m_p = np.zeros(clip_len, np.float32)
    y_c = np.zeros(clip_len, np.float32)
    m_c = np.ones(clip_len, np.float32)

    events_sorted = sorted(events, key=lambda x: x['label_idx'])

    # Fill y_c piecewise from carry_in through each event
    state = carry_in
    seg_start = 0
    for e in events_sorted:
        t = e['label_idx']
        label = e['label']

        if seg_start < t:
            y_c[seg_start:t] = float(state)

        state = (label == 'touch')  # new state immediately after this event
        y_c[t] = float(state)
        seg_start = t + 1

    if seg_start < clip_len:
        y_c[seg_start:] = float(state)

    # Fill y_r (with shoulders), y_p, m_p for every event
    for e in events_sorted:
        t = e['label_idx']
        label = e['label']

        for delta, weight in _SHOULDER.items():
            ti = t + delta
            if 0 <= ti < clip_len:
                y_r[ti] = max(y_r[ti], weight)

        m_p[t] = 1.0
        y_p[t] = 1.0 if label == 'touch' else 0.0

    return y_r, m_r, y_p, m_p, y_c, m_c


class ActionSpotDatasetCT(ActionSpotDataset):
    """
    Training/validation CT dataset.

    Overrides _store_clips / _load_clips / _get_one from ActionSpotDataset.
    All other behaviour (hand handler, frame reader, augmentation) is inherited.
    """

    def _ct_store_path(self):
        return os.path.join(
            self._store_dir,
            f'LEN{self._clip_len}CT_SPLIT{self._split}',
        )

    def _store_clips(self):
        self._frame_paths = []
        self._ct_meta = []  # list of {'carry_in': bool, 'events': [...]}

        for video in tqdm(self._labels):
            video_len = int(video['num_frames'])
            events_sorted = sorted(video['events'], key=lambda e: e['frame'])

            for base_idx in range(
                -self._pad_len * self._stride,
                max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride),
                self._overlap,
            ):
                clip_end_frame = base_idx + self._clip_len * self._stride

                # Contact state just before this clip window
                carry_in = False
                for e in events_sorted:
                    if e['frame'] < base_idx:
                        carry_in = (e['label'] == 'touch')
                    else:
                        break

                # Events whose frame falls within [base_idx, clip_end_frame)
                in_window = []
                for e in events_sorted:
                    if base_idx <= e['frame'] < clip_end_frame:
                        label_idx = (e['frame'] - base_idx) // self._stride
                        if 0 <= label_idx < self._clip_len:
                            in_window.append({
                                'label': e['label'],
                                'label_idx': label_idx,
                            })

                frames_paths = self._frame_reader.load_paths(
                    video['video'], base_idx, clip_end_frame,
                    stride=self._stride,
                )

                if frames_paths[1] != -1:
                    self._frame_paths.append(frames_paths)
                    self._ct_meta.append({
                        'carry_in': carry_in,
                        'events': in_window,
                    })

        store_path = self._ct_store_path()
        os.makedirs(store_path, exist_ok=True)
        with open(os.path.join(store_path, 'frame_paths.pkl'), 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(os.path.join(store_path, 'ct_meta.pkl'), 'wb') as f:
            pickle.dump(self._ct_meta, f)
        print(f'[INFO] Stored CT clips to {store_path}')

    def _load_clips(self):
        store_path = self._ct_store_path()
        with open(os.path.join(store_path, 'frame_paths.pkl'), 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(os.path.join(store_path, 'ct_meta.pkl'), 'rb') as f:
            self._ct_meta = pickle.load(f)
        print(f'[INFO] Loaded CT clips from {store_path}')

    def _get_one(self):
        idx = random.randint(0, self._total_len - 1)
        frames_path = self._frame_paths[idx]
        meta = self._ct_meta[idx]

        frames, hands = self._frame_reader.load_frames(
            frames_path, pad=True, stride=self._stride)

        y_r, m_r, y_p, m_p, y_c, m_c = compute_ct_labels(
            self._clip_len, meta['carry_in'], meta['events'],
        )

        return {
            'frame': frames,
            'hands': hands,
            'contains_event': int(np.any(y_r > 0)),
            'y_r': torch.from_numpy(y_r),
            'm_r': torch.from_numpy(m_r),
            'y_p': torch.from_numpy(y_p),
            'm_p': torch.from_numpy(m_p),
            'y_c': torch.from_numpy(y_c),
            'm_c': torch.from_numpy(m_c),
        }

    def __getitem__(self, unused):
        return self._hand_handler(self._get_one())
