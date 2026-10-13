#!/bin/bash
set -e
HAMMER=/path/to/hammer                       # folder that holds scene*_traj*_* folders
BOOSTER=/path/to/booster/train/balanced      # Booster labeled split (never used for training)
mkdir -p ckpt results

python train.py --root $HAMMER --mode teacher --out ckpt/teacher.pt
python train.py --root $HAMMER --mode student --w_kd 0 --w_pol 0 --out ckpt/baseline.pt
python train.py --root $HAMMER --mode student --teacher ckpt/teacher.pt --out ckpt/ours.pt
python train.py --root $HAMMER --mode student --teacher ckpt/teacher.pt --w_pol 0 --out ckpt/kd_only.pt
python train.py --root $HAMMER --mode student --w_kd 0 --out ckpt/pol_only.pt

for m in baseline ours kd_only pol_only; do
  python eval.py --data booster --root $BOOSTER --ckpt ckpt/$m.pt --out results/booster_$m.json
  python eval.py --data hammer --root $HAMMER --ckpt ckpt/$m.pt --out results/hammer_$m.json
done
python eval.py --data hammer --root $HAMMER --ckpt ckpt/teacher.pt --polar --out results/hammer_teacher.json
