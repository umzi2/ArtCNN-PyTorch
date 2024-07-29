import torch
import torch.nn as nn

from neosr.utils.registry import ARCH_REGISTRY
from neosr.archs.arch_util import net_opt

upscale, __ = net_opt()


class DepthToSpace(nn.Module):
    def __init__(self, filters: int, out_ch: int, kernel_size: int, scale: int):
        super(DepthToSpace, self).__init__()

        self.upscale = nn.Sequential(
            nn.Conv2d(filters, out_ch * (scale ** 2), kernel_size, 1, kernel_size // 2),
            nn.PixelShuffle(scale))

    def forward(self, x):
        return self.upscale(x)


class ActConv(nn.Sequential):
    def __init__(self, filters: int, kernel_size: int, act: nn.Module = nn.ReLU):
        super().__init__(
            nn.Conv2d(filters, filters, kernel_size, 1, kernel_size // 2),
            act()
        )


class ResBlock(nn.Module):
    def __init__(self, filters: int, kernel_size: int, act: nn.Module = nn.ReLU):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            ActConv(filters, kernel_size, act),
            ActConv(filters, kernel_size, act),
            nn.Conv2d(filters, filters, kernel_size, 1, kernel_size // 2)
        )

    def forward(self, x):
        res = self.conv(x)
        return x + res


@ARCH_REGISTRY.register()
class ArtCNN(nn.Module):
    def __init__(self, in_ch: int = 3, scale: int = upscale, filters: int = 96, n_block=15, kernel_size: int = 3,
                 act: nn.Module = nn.ReLU):
        super(ArtCNN, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, filters, kernel_size, 1, kernel_size // 2)
        self.res_block = nn.Sequential(*[ResBlock(filters, kernel_size, act) for _ in range(n_block)] + [
            nn.Conv2d(filters, filters, kernel_size, 1, kernel_size // 2)])
        self.depth_to_space = DepthToSpace(filters, in_ch, kernel_size, scale)

    def forward(self, x):
        x = self.conv0(x)
        x = self.res_block(x) + x
        return self.depth_to_space(x)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath("."))
    upscale = 4
    height = 256
    width = 256
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = ArtCNN(
    ).cuda()

    params = sum(map(lambda x: x.numel(), model.parameters()))
    results = dict()
    results["runtime"] = []
    model.eval()

    x = torch.randn((1, 3, height, width)).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # x = model(x)
        for _ in range(10):
            x_sr = model(x)
        for _ in range(100):
            start.record()
            x_sr = model(x)
            end.record()
            torch.cuda.synchronize()
            results["runtime"].append(start.elapsed_time(end))  # milliseconds
    print(x.shape)

    print("{:.2f}ms".format(sum(results["runtime"]) / len(results["runtime"])))
    results["memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    print("Max Memery:{:.2f}[M]".format(results["memory"]))
    print(f"Height:{height}->{x_sr.shape[2]}\nWidth:{width}->{x_sr.shape[3]}\nParameters:{params / 1e3:.2f}K")
