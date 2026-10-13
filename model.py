import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def cbr(c):
    return [nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)]


class DySample(nn.Module):
    def __init__(self, c, g=4):
        super().__init__()
        self.g = g
        self.offset = nn.Conv2d(c, 8 * g, 1)
        self.scope = nn.Conv2d(c, 8 * g, 1, bias=False)
        nn.init.normal_(self.offset.weight, std=1e-3)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.scope.weight)
        h = torch.tensor([-0.25, 0.25])
        self.register_buffer('init_pos', torch.stack(torch.meshgrid(h, h, indexing='ij')).transpose(1, 2).repeat(1, g, 1).reshape(1, -1, 1, 1))

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.float()
        off = (self.offset(x) * self.scope(x).sigmoid() * 0.5).float() + self.init_pos
        gx, gy = torch.meshgrid(torch.arange(W, device=x.device) + 0.5, torch.arange(H, device=x.device) + 0.5, indexing='ij')
        grid = torch.stack([gx, gy]).transpose(1, 2)[None, :, None] + off.view(B, 2, -1, H, W)
        grid = 2 * grid / torch.tensor([W, H], device=x.device).view(1, 2, 1, 1, 1) - 1
        grid = F.pixel_shuffle(grid.reshape(B, -1, H, W), 2).view(B, 2, -1, 2 * H, 2 * W).permute(0, 2, 3, 4, 1).flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.g, -1, H, W), grid, align_corners=False, padding_mode='border').view(B, -1, 2 * H, 2 * W)


class DepthNet(nn.Module):
    def __init__(self, backbone, polar=False, tune_blocks=4, c=256):
        super().__init__()
        self.bb = AutoModel.from_pretrained(backbone)
        cfg = self.bb.config
        n, D, self.ps, self.k = cfg.num_hidden_layers, cfg.hidden_size, cfg.patch_size, 1 + cfg.num_register_tokens
        self.taps = [n // 4 * (i + 1) - 1 for i in range(4)]
        self.bb.requires_grad_(False)
        for m in [*self.bb.model.layer[n - tune_blocks:], self.bb.norm]:
            m.requires_grad_(True)
        self.readout = nn.ModuleList(nn.Sequential(nn.Linear(2 * D, D), nn.GELU()) for _ in self.taps)
        self.proj = nn.ModuleList(nn.Sequential(nn.Linear(D, c, bias=False), nn.GELU()) for _ in self.taps)
        self.tap_w = nn.Parameter(torch.ones(len(self.taps)))
        self.sde = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.BatchNorm2d(c))
        self.up = nn.Sequential(*[m for _ in range(2) for m in (DySample(c), *cbr(c), DySample(c), *cbr(c), *cbr(c))])
        self.out = nn.Sequential(nn.Conv2d(c, c // 2, 3, padding=1), nn.ReLU(True), nn.Conv2d(c // 2, 32, 3, padding=1), nn.ReLU(True), nn.Conv2d(32, 1, 1))
        self.kd = nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))
        self.pol_head = nn.Conv2d(c, 2, 1)
        self.pol = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, D, self.ps, stride=self.ps)) if polar else None
        if polar:
            nn.init.zeros_(self.pol[-1].weight)
            nn.init.zeros_(self.pol[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.bb.eval()
        return self

    def forward(self, x, pol=None):
        B, _, H, W = x.shape
        h = self.bb.embeddings(x)
        if pol is not None:
            h = torch.cat([h[:, :self.k], h[:, self.k:] + self.pol(pol).flatten(2).transpose(1, 2)], 1)
        pe = self.bb.rope_embeddings(x)
        w, t = self.tap_w.softmax(0), 0
        for i, blk in enumerate(self.bb.model.layer[:self.taps[-1] + 1]):
            h = blk(h, position_embeddings=pe)
            if i in self.taps:
                z = self.bb.norm(h)
                p, cls = z[:, self.k:], z[:, :1].expand(-1, z.shape[1] - self.k, -1)
                j = self.taps.index(i)
                t = t + w[j] * self.proj[j](self.readout[j](torch.cat([p, cls], -1)))
        f = t.transpose(1, 2).reshape(B, -1, H // self.ps, W // self.ps)
        f = F.relu(f + self.sde(f))
        return self.out(self.up(f))[:, 0], f
