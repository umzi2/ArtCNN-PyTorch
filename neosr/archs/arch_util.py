import collections.abc
import functools
import inspect
from collections.abc import Mapping
from itertools import repeat
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch
from torch import nn
from torch.nn import functional as F

from neosr.utils.options import parse_options


def net_opt():
    # initialize options parsing
    root_path = Path(__file__).parents[2]
    opt, args = parse_options(root_path, is_train=True)

    # set variable for scale factor and training phase
    # conditions needed due to convert.py

    if args.input is None:
        upscale = opt["scale"]
        if "train" in opt["datasets"]:
            training = True
        else:
            training = False
    else:
        upscale = args.scale
        training = False

    return upscale, training


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ):
        super().__init__()

        try:
            assert in_channels >= groups and in_channels % groups == 0
        except:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample.
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        __, training = net_opt()
        self.training = training

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def store_neosr_defaults(*, extra_parameters: Mapping[str, object] = {}):
    """
    Stores the neosr default hyperparameters in a `neosr_params` attribute.
    Based on Spandrel implementation (MIT license):
        https://github.com/chaiNNer-org/spandrel
    """

    def get_arg_defaults(spec: inspect.FullArgSpec) -> dict[str, Any]:
        defaults = {}
        if spec.kwonlydefaults is not None:
            defaults = spec.kwonlydefaults

        if spec.defaults is not None:
            defaults = {
                **defaults,
                **dict(
                    zip(spec.args[-len(spec.defaults) :], spec.defaults, strict=False)
                ),
            }

        return defaults

    class WithHyperparameters(Protocol):
        neosr_params: dict[str, Any]

    C = TypeVar("C", bound=WithHyperparameters)

    def inner(cls: type[C]) -> type[C]:
        old_init = cls.__init__

        spec = inspect.getfullargspec(old_init)
        defaults = get_arg_defaults(spec)

        @functools.wraps(old_init)
        def new_init(self: C, **kwargs):
            # remove extra parameters from kwargs
            for k, v in extra_parameters.items():
                if k in kwargs:
                    if kwargs[k] != v:
                        raise ValueError(
                            f"Expected hyperparameter {k} to be {v}, but got {kwargs[k]}"
                        )
                    del kwargs[k]

            self.hyperparameters = {**extra_parameters, **defaults, **kwargs}
            old_init(self, **kwargs)

        cls.__init__ = new_init
        return cls

    return inner


# From PyTorch
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple