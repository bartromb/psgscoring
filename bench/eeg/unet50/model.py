"""1D-U-Net met kernel 21, naar Ehrlich e.a. 2024 (eigen implementatie, BSD-vrij).

Encoder 16-32-64-128-256 met poolingfactoren (2, 4, 4, 4) -> 128x downsampling
(0,39 Hz in de bottleneck, receptief veld ~2 min); decoder spiegelbeeldig met
lineaire upsampling en skip-concatenatie; acht dubbele convoluties in
encoder+decoder plus een invoerblok, BatchNorm+ReLU, geen dropout. Uitvoer:
één logit per invoersample (50 Hz).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, k=21):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=p, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=p, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch=3, base=16, pools=(2, 4, 4, 4), k=21):
        super().__init__()
        self.pools = pools
        chs = [base * 2 ** i for i in range(len(pools) + 1)]   # 16..256
        self.inc = DoubleConv(in_ch, chs[0], k)
        self.downs = nn.ModuleList([DoubleConv(chs[i], chs[i + 1], k) for i in range(len(pools))])
        self.ups = nn.ModuleList([DoubleConv(chs[i + 1] + chs[i], chs[i], k) for i in range(len(pools))])
        self.outc = nn.Conv1d(chs[0], 1, 1)
        self.factor = 1
        for p in pools:
            self.factor *= p
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        n = x.shape[-1]
        pad = (-n) % self.factor
        if pad:
            x = F.pad(x, (0, pad))
        skips = []
        h = self.inc(x)
        for p, d in zip(self.pools, self.downs):
            skips.append(h)
            h = d(F.max_pool1d(h, p))
        for p, u in zip(reversed(self.pools), reversed(self.ups)):
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-1], mode="linear", align_corners=False)
            h = u(torch.cat([s, h], dim=1))
        out = self.outc(h)[:, 0]
        return out[..., :n]
