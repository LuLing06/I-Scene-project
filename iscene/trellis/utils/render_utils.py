import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..renderers import GaussianRenderer, MeshRenderer
from ..representations import Gaussian, MeshExtractResult
from .random_utils import sphere_hammersley_sequence


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        # Prefer explicit width/height if provided; fall back to square resolution.
        renderer.rendering_options.width = kwargs.get('width', None)
        renderer.rendering_options.height = kwargs.get('height', None)
        renderer.rendering_options.resolution = kwargs.get('resolution', 512 if (kwargs.get('width', None) is None or kwargs.get('height', None) is None) else None)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        else:
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
    return rets

def render_multiview(sample, resolution=512, nviews=30, r=2, verbose=True):
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(
        sample,
        extrinsics,
        intrinsics,
        {'resolution': resolution, 'bg_color': (0, 0, 0)},
        verbose=verbose,
    )
    return res['color'], extrinsics, intrinsics
