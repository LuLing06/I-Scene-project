import torch
import torch.nn as nn
from .. import SparseTensor
from .. import DEBUG
from . import SPCONV_ALGO


def _lexsort_coords(coords: torch.Tensor) -> torch.Tensor:
    if coords.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=coords.device)

    coords64 = coords.to(torch.int64)
    max_vals = coords64.max(dim=0).values + 1
    key = coords64[:, 0]
    for dim in range(1, coords64.shape[1]):
        key = key * max_vals[dim] + coords64[:, dim]
    return torch.argsort(key)


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        algo = None
        if SPCONV_ALGO == 'native':
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == 'implicit_gemm':
            algo = spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1 and (padding is None):
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
        else:
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
        new_data = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # Keep sparse points in a canonical order so later sparse attention does
            # not depend on backend-specific output ordering.
            fwd = _lexsort_coords(new_data.indices)
            bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
            out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)

        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride)
        if spatial_changed:
            # recover the original spconv order
            data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
            bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
            data = data.replace_feature(x.feats[bwd])
            if DEBUG:
                assert torch.equal(data.indices, x.coords[bwd]), 'Recover the original order failed'
        else:
            data = x.data

        new_data = self.conv(data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout
        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out
