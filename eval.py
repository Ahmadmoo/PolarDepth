import argparse
import json

import cv2
import numpy as np
import torch

from data import Hammer, booster_files, read_booster, to_tensor
from model import DepthNet

pa = argparse.ArgumentParser()
pa.add_argument('--ckpt', required=True)
pa.add_argument('--data', choices=['booster', 'hammer'], default='booster')
pa.add_argument('--root', required=True)
pa.add_argument('--polar', action='store_true')
pa.add_argument('--backbone', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
pa.add_argument('--size', type=int, nargs=2, default=[480, 640])
pa.add_argument('--out')
args = pa.parse_args()
assert not (args.polar and args.data == 'booster'), 'Booster has no polarization: evaluate students only'

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = DepthNet(args.backbone, polar=args.polar).to(dev).eval()
net.load_state_dict(torch.load(args.ckpt, map_location=dev))


@torch.no_grad()
def run(x, pol=None):
    with torch.autocast(dev, torch.bfloat16):
        return net(x[None].to(dev), None if pol is None else pol[None].to(dev))[0][0].float().cpu().numpy()


def evaluate(pred, disp, C, cat):
    v = disp > 0
    p = (pred - pred[v].min()) / (pred[v].max() - pred[v].min())
    s, b = np.polyfit(p[v], disp[v], 1)
    with np.errstate(divide='ignore'):
        d = C / (p * s + b)
    d[np.isinf(d)] = 1e4
    d = np.clip(d, 1, 1e4)
    g = np.where(v, C / np.where(v, disp, 1), 0)
    out = {}
    groups = [('All', v)] + ([('ToM', v & np.isin(cat, [2, 3])), ('Other', v & np.isin(cat, [0, 1]))] if cat is not None else [])
    for k, sel in groups:
        if sel.any():
            r, e = np.maximum(d[sel] / g[sel], g[sel] / d[sel]), np.abs(d[sel] - g[sel])
            out[k] = {**{f'd{t:.2f}': 100 * (r < t).mean() for t in (1.05, 1.10, 1.15, 1.20, 1.25)},
                      'absrel': (e / g[sel]).mean(), 'mae': e.mean(), 'rmse': np.sqrt((e ** 2).mean())}
    return out


results = []
if args.data == 'booster':
    for f in booster_files(args.root):
        rgb, disp, cat, fx, base = read_booster(f)
        disp = cv2.resize(disp, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST) * 0.25
        fx *= 0.25
        disp[(disp > fx * base) | (disp < fx * base / 1e4)] = 0
        cat = None if cat is None else cv2.resize(cat.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        H, W = disp.shape
        x = to_tensor(cv2.resize(rgb, (W // net.ps * net.ps, H // net.ps * net.ps), interpolation=cv2.INTER_AREA))
        results.append(evaluate(cv2.resize(run(x), (W, H), interpolation=cv2.INTER_LINEAR), disp, fx * base, cat))
else:
    ds = Hammer(args.root, range(12, 15), args.size)
    for i in range(len(ds)):
        x, pol, depth = ds[i]
        depth = depth[0].numpy() * 1000
        disp = np.where((depth > 1) & (depth < 1e4), 1 / np.maximum(depth, 1), 0)
        results.append(evaluate(run(x, pol if args.polar else None), disp, 1.0, None))

summary = {k: {m: float(np.mean([r[k][m] for r in results if k in r])) for m in next(r[k] for r in results if k in r)}
           for k in ['All', 'ToM', 'Other'] if any(k in r for r in results)}
print(f'{args.ckpt} on {args.data} ({len(results)} images; depth errors in mm)')
print(f'{"":6s}' + ''.join(f'{m:>9s}' for m in summary['All']))
for k, v in summary.items():
    print(f'{k:6s}' + ''.join(f'{x:9.3f}' for x in v.values()))
if args.out:
    json.dump(summary, open(args.out, 'w'), indent=1)
