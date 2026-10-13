import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import Hammer
from model import DepthNet

pa = argparse.ArgumentParser()
pa.add_argument('--root', required=True)
pa.add_argument('--out', required=True)
pa.add_argument('--mode', choices=['teacher', 'student'], default='student')
pa.add_argument('--teacher')
pa.add_argument('--w_kd', type=float, default=1.0)
pa.add_argument('--w_pol', type=float, default=0.5)
pa.add_argument('--backbone', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
pa.add_argument('--epochs', type=int, default=20)
pa.add_argument('--bs', type=int, default=4)
pa.add_argument('--lr', type=float, default=1e-4)
pa.add_argument('--size', type=int, nargs=2, default=[480, 640])
pa.add_argument('--workers', type=int, default=8)
pa.add_argument('--seed', type=int, default=0)
args = pa.parse_args()
student = args.mode == 'student'
w_kd, w_pol = (args.w_kd, args.w_pol) if student else (0, 0)


def norm_disp(d, m):
    out = torch.zeros_like(d)
    for i in range(d.shape[0]):
        v = d[i][m[i]]
        t = v.median()
        out[i] = (d[i] - t) / (v - t).abs().mean().clamp_min(1e-6)
    return out


def depth_loss(pred, gt, m):
    r, mf = norm_disp(pred, m) - norm_disp(gt, m), m.float()
    loss = (r.abs() * mf).sum() / mf.sum()
    for s in range(4):
        rs, ms = r[:, ::2 ** s, ::2 ** s], mf[:, ::2 ** s, ::2 ** s]
        gx = (rs[:, :, 1:] - rs[:, :, :-1]).abs() * ms[:, :, 1:] * ms[:, :, :-1]
        gy = (rs[:, 1:] - rs[:, :-1]).abs() * ms[:, 1:] * ms[:, :-1]
        loss = loss + 0.5 * (gx.sum() + gy.sum()) / ms.sum().clamp_min(1)
    return loss


def kd_loss(fs, ft, rho):
    w = rho / rho.mean((2, 3), keepdim=True).clamp_min(1e-6)
    return ((1 - F.cosine_similarity(fs, ft, dim=1)) * w[:, 0]).mean()


torch.manual_seed(args.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dl = DataLoader(Hammer(args.root, range(2, 12), args.size, train=True), args.bs, shuffle=True, num_workers=args.workers,
                drop_last=True, pin_memory=dev == 'cuda', generator=torch.Generator().manual_seed(args.seed))
net = DepthNet(args.backbone, polar=not student).to(dev)
teacher = None
if w_kd > 0:
    teacher = DepthNet(args.backbone, polar=True).to(dev).eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(args.teacher, map_location=dev))

named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
opt = torch.optim.AdamW([{'params': [p for n, p in named if n.startswith('bb.')], 'lr': args.lr * 0.1},
                         {'params': [p for n, p in named if not n.startswith('bb.')], 'lr': args.lr}], weight_decay=0.01)
total = args.epochs * len(dl)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * s / total)))

step = 0
for ep in range(args.epochs):
    net.train()
    for rgb, pol, depth in dl:
        rgb, pol, depth = rgb.to(dev, non_blocking=True), pol.to(dev, non_blocking=True), depth.to(dev, non_blocking=True)[:, 0]
        m = depth > 1e-3
        gt = torch.where(m, 1 / depth.clamp_min(1e-3), torch.zeros_like(depth))
        with torch.autocast(dev, torch.bfloat16):
            pred, f = net(rgb, None if student else pol)
            loss = depth_loss(pred.float(), gt, m)
            if w_kd > 0:
                with torch.no_grad():
                    ft = teacher(rgb, pol)[1]
                loss = loss + w_kd * kd_loss(net.kd(f).float(), ft.float(), F.avg_pool2d(pol.norm(dim=1, keepdim=True), net.ps))
            if w_pol > 0:
                loss = loss + w_pol * F.l1_loss(net.pol_head(f).float(), F.avg_pool2d(pol, net.ps))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            print(f'epoch {ep} step {step}/{total} loss {loss.item():.4f}', flush=True)
    torch.save(net.state_dict(), args.out)
