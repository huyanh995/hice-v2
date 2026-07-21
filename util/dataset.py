import os

from util.io import load_text


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}

def read_fps(video_frame_dir):
    with open(os.path.join(video_frame_dir, 'fps.txt')) as fp:
        return float(fp.read())

def infer_backbone_grid_size(crop_dim):
    """rny00x (regnety) backbones downsample by a fixed total stride of 32 --
    verified empirically (224->7, 384->12, 448->14), so the obj_head heatmap's
    native grid is just crop_dim // 32."""
    assert crop_dim % 32 == 0, (
        f'crop_dim ({crop_dim}) must be a multiple of 32 to match the rny backbone\'s '
        f'stride-32 downsampling exactly.')
    return crop_dim // 32
