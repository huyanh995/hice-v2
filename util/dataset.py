import os

from util.io import load_text


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}

def read_fps(video_frame_dir):
    with open(os.path.join(video_frame_dir, 'fps.txt')) as fp:
        return float(fp.read())

def infer_backbone_grid_size(feature_arch):
    """Native spatial grid size (Gh=Gw) this backbone's feature map has, before any
    pooling. rny family is a fixed 7x7 (regnet's own stride reduction). vjepa2_1
    family is native_res // patch_size -- patch_size=16 for every vjepa2_1_vit_*_384
    hub variant (base/large/giant/gigantic all share it, see backbones.py).

    Used to size obj_head's target/heatmap grid (dataset/frame.py's render_obj_target)
    to match whatever backbone is actually selected, computed here (dataset-prep time,
    before the model is constructed) rather than duplicating model.py's own encoder-
    attribute-based computation of the same value.
    """
    if feature_arch.startswith('vjepa2_1_vit_'):
        native_res = int(feature_arch.rsplit('_', 1)[-1])
        return native_res // 16
    return 7
