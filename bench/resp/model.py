"""1D-U-Net (kernel 21) met n invoerkanalen en n uitgangskoppen — eigen implementatie, BSD-vrij;
zelfde ontwerp als bench/eeg/unet50/model.py met `in_ch` en `out_ch` als parameters."""
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
    def __init__(self, in_ch=5, out_ch=2, base=16, pools=(2, 4, 4, 4), k=21):
        super().__init__()
        self.pools = pools
        chs = [base * 2 ** i for i in range(len(pools) + 1)]
        self.inc = DoubleConv(in_ch, chs[0], k)
        self.downs = nn.ModuleList([DoubleConv(chs[i], chs[i + 1], k) for i in range(len(pools))])
        self.ups = nn.ModuleList([DoubleConv(chs[i + 1] + chs[i], chs[i], k) for i in range(len(pools))])
        self.outc = nn.Conv1d(chs[0], out_ch, 1)
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
        return self.outc(h)[..., :n]
