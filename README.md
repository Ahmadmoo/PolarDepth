# PolarDepth

Train with polarization, test with RGB only.
A depth network for transparent and mirror surfaces. During training it learns from a teacher that sees polarization. At test time it needs only an RGB image.

---

## 1. Idea

- On glass and mirrors, RGB carries little geometry. Polarization does: by the Fresnel equations, reflected light is polarized according to the surface orientation.
- **Teacher**: RGB + polarization → depth.
- **Student**: RGB → depth. It learns from ground-truth depth and from the teacher's features.
- At test time only the student runs. No polarization camera.

---

## 2. Choices

| Part | Choice | Reason |
|---|---|---|
| Train data | HAMMER, scenes 2–11 (official train split) | Pixel-aligned RGB + polarization + dense rendered GT depth, with transparent and reflective objects |
| Test data | Booster labeled split (228 images, 38 scenes), never used for training | Dataset of the NTIRE HR Depth from Specular and Transparent Surfaces challenges; has GT and ToM material masks. NTIRE val/test GT is hidden |
| In-domain check | HAMMER scenes 12–14 | Sanity check, and the only place the teacher can be tested (needs polarization) |
| Backbone | DINOv3 ViT-B/16 (LVD-1689M), last 4 blocks trained | Strong dense features; cheap to fine-tune |
| Decoder | SDT decoder from AnyDepth (DINOv3 + SDT) | 85–89% fewer parameters than DPT, simple single-path design |
| Output | Affine-invariant disparity | Matches the Booster mono protocol (scale + shift alignment) |
| Polarization signal | Normalized linear Stokes vector `s = (S1/S0, S2/S0) = DoLP·(cos 2·AoLP, sin 2·AoLP)` | See 3.1 |
| Transfer | DoLP-weighted predictive feature distillation + polarization-vector prediction | See 3.3 |
| Baseline | Same student, same init, same data order, same schedule, transfer losses off | Only the transfer losses differ |

---

## 3. Method

### 3.1 Polarization signal

From the four polarizer images `I0, I45, I90, I135`:

```
S0 = (I0 + I45 + I90 + I135) / 2
S1 = I0 - I90
S2 = I45 - I135
s  = (S1, S2) / S0        # 2 channels, clipped to the unit disk
```

- `|s|` is DoLP, `atan2(s2, s1) / 2` is AoLP.
- No angle wrap problem (AoLP is periodic, `s` is not).
- Where DoLP is low, AoLP is noise. `s` goes to zero there by itself, so no extra masking is needed.
- Dark pixels (`S0 < 0.02`) are set to 0.
- A horizontal flip mirrors AoLP, so the loader flips the sign of `s2`.

### 3.2 Network

```
RGB ─► DINOv3 ViT-B/16 ─► layers 2,5,8,11 ─► SDT decoder ─► disparity (full res)
                                   │
                                   └─► fused map f (256 ch, 1/16 res)
```

- **SDT decoder**: per-layer CLS readout + projection, softmax-weighted fusion, depthwise detail enhancer, two ×4 DySample stages, output convs.
- **Teacher**: same network plus a polarization stem: `conv3×3 → GELU → 16×16 stride-16 conv`. Its output is added to the patch tokens before block 1. The last conv starts at zero, so the teacher starts as the RGB model.
- Blocks 1–8 of DINOv3 are frozen. Blocks 9–12, the final norm and the decoder are trained. The backbone stays in eval mode (no RoPE jitter).

### 3.3 Losses

Depth loss (all models): scale-and-shift-invariant L1 on disparity (median / mean-absolute-deviation normalization) + 0.5 × gradient matching at 4 scales.

Student:

```
L = L_depth + λ_kd · L_kd + λ_pol · L_pol

L_kd  = mean_j  w_j · (1 − cos( q(f_S)_j , f_T_j ))      # f_T from the frozen teacher
w_j   = DoLP_j / mean(DoLP)                              # DoLP averaged in each 16×16 patch
L_pol = | g(f_S) − avgpool16(s) |_1
```

- `q` is a 2-layer 1×1 conv predictor, `g` a 1×1 conv. Both are dropped at test time.
- `L_kd`: the student predicts the teacher's polarization-aware features. The weight `w` puts the transfer where polarization carries signal (specular and transparent regions) and ignores patches where DoLP is zero.
- `L_pol`: the student predicts the polarization vector from RGB. This is a direct physical target that works even if the teacher learns to rely on polarization only a little.
- Defaults: `λ_kd = 1`, `λ_pol = 0.5`.

### 3.4 Training

AdamW, lr 1e-4 (decoder) and 1e-5 (backbone), weight decay 0.01, 500-step warmup, cosine decay, 20 epochs, batch 4, 480×640, bf16 autocast.

---

## 4. Setup

```bash
pip install -r requirements.txt
hf auth login        # DINOv3 weights are gated: first request access at
                     # https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
```

---

## 5. Data

### 5.1 HAMMER (training)

```bash
wget http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip
unzip _dataset_processed.zip
```

`--root` is the folder that holds the `scene*_traj*_*` folders. Used files:

```
sceneXX_trajY_Z/polarization/rgb/000000.png    # RGB
sceneXX_trajY_Z/polarization/pol/000000.png    # 2×2 grid of the 4 polarizer images
sceneXX_trajY_Z/polarization/_gt/000000.png    # rendered depth, uint16, mm
```

`*_naked_*` sequences (no objects) are skipped.

### 5.2 Booster (test only)

Download `booster_gt.zip` (19 GB) from [AMS Acta](https://amsacta.unibo.it/id/eprint/6876), or the NTIRE 2026 training data from the [challenge page](https://cvlab.unibo.it/booster-web/ntire26.html) (same 228 images). `--root` is `.../train/balanced`. Used files:

```
<Scene>/camera_00/im*.png      # left images, several lights
<Scene>/disp_00.npy            # GT disparity
<Scene>/mask_cat.png           # material classes: 0,1 other, 2,3 transparent or mirror (ToM)
<Scene>/calib_00-02.xml        # proj_matL (fx), baselineLR (mm)
```

---

## 6. Run

Set the two paths in `run.sh`, then:

```bash
bash run.sh
```

It trains the teacher, the baseline, the full student and two ablations, then evaluates them:

| Run | `--mode` | `λ_kd` | `λ_pol` |
|---|---|---|---|
| teacher | teacher | – | – |
| baseline | student | 0 | 0 |
| ours | student | 1 | 0.5 |
| kd_only | student | 1 | 0 |
| pol_only | student | 0 | 0.5 |

Lower `--bs` if you run out of GPU memory.

---

## 7. Evaluation protocol

Same as the Booster mono benchmark and the [Depth4ToM](https://github.com/CVLAB-Unibo/Depth4ToM-code) evaluation code:

1. GT disparity at 1/4 resolution (1028×752); valid depth range 1–10000 mm.
2. Prediction min-max normalized, then least-squares scale and shift to GT disparity.
3. Depth = baseline · fx / disparity, clipped to [1, 10000] mm.
4. Metrics: δ < 1.05 / 1.10 / 1.15 / 1.20 / 1.25 (%), AbsRel, MAE (mm), RMSE (mm).
5. Reported on All, ToM and Other pixels, averaged over images.

HAMMER uses the same steps with its metric GT (in mm) at the training resolution.

The NTIRE 2026 mono track scores metric depth on hidden GT. This repo uses the labeled split with the affine-invariant protocol, so the numbers are not comparable to that leaderboard.

Results table to fill:

| Model | Test input | Booster ToM δ1.05 | ToM AbsRel | All δ1.05 | All AbsRel |
|---|---|---|---|---|---|
| baseline | RGB | | | | |
| kd_only | RGB | | | | |
| pol_only | RGB | | | | |
| ours | RGB | | | | |

---

## 8. Checks before a long run

**Polarizer order in HAMMER `pol/*.png`.** The loader assumes (top-left, top-right, bottom-left, bottom-right) = (0°, 45°, 90°, 135°), set by `POL_LAYOUT` in `data.py`. This follows the PPFT code, which marks it as unverified. For a correct pairing, `I0 + I90 = I45 + I135`. Check it:

```python
import cv2, glob, numpy as np
f = sorted(glob.glob('/path/to/hammer/scene2_traj1_1/polarization/pol/*.png'))[0]
im = cv2.imread(f, -1).astype(np.float32); im = im[..., :3].mean(-1) if im.ndim == 3 else im
h, w = im.shape[0] // 2, im.shape[1] // 2
q = [im[:h, :w], im[:h, w:], im[h:, :w], im[h:, w:]]
for name, (a, b) in {'TL+BL vs TR+BR (assumed)': ((0, 2), (1, 3)), 'TL+BR vs TR+BL': ((0, 3), (1, 2)), 'TL+TR vs BL+BR': ((0, 1), (2, 3))}.items():
    print(name, np.abs(q[a[0]] + q[a[1]] - q[b[0]] - q[b[1]]).mean() / np.mean(sum(q) / 2))
```

The assumed line should give the smallest value. If another line wins, change `POL_LAYOUT`. Swapping 0°↔90° or 45°↔135° only flips the sign of `s1` or `s2`; training does not care about that.

**Linearity.** Stokes math assumes linear intensities. The loader assumes the polarization PNGs have no gamma curve.

---

## 9. Files

| File | Content |
|---|---|
| `data.py` | HAMMER loader, Stokes vector, Booster reader |
| `model.py` | DINOv3 + SDT decoder, polarization stem, predictor and polarization head |
| `train.py` | Teacher and student training, losses |
| `eval.py` | Booster and HAMMER evaluation |
| `run.sh` | All runs |

---

## 10. Status

All scripts were run end to end (teacher, baseline, student, both evaluations) on synthetic data written in the exact HAMMER and Booster file formats, with a small random DINOv3. The Stokes code recovers known DoLP/AoLP, the metric code gives perfect scores for a perfect prediction, and the baseline's transfer modules stay untouched. It has not yet been run on the real datasets.

---

## References

- HAMMER: Jung et al., *On the Importance of Accurate Geometry Data for Dense 3D Vision Tasks*, CVPR 2023. [code/data](https://github.com/Junggy/HAMMER-dataset)
- Booster: Zama Ramirez et al., *Booster: a Benchmark for Depth from Images of Specular and Transparent Surfaces*, TPAMI 2024. [site](https://cvlab-unibo.github.io/booster-web/)
- DINOv3: Siméoni et al., 2025. [code](https://github.com/facebookresearch/dinov3)
- AnyDepth (SDT decoder): Ren et al., *AnyDepth: Depth Estimation Made Easy*, arXiv:2601.02760. [code](https://github.com/AIGeeksGroup/AnyDepth)
- PPFT: Ikemura et al., *Robust Depth Enhancement via Polarization Prompt Fusion Tuning*, CVPR 2024. [code](https://github.com/lastbasket/Polarization-Prompt-Fusion-Tuning)
- Depth4ToM: Costanzino et al., *Learning Depth Estimation for Transparent and Mirror Surfaces*, ICCV 2023. [code](https://github.com/CVLAB-Unibo/Depth4ToM-code)

Data licenses: HAMMER MIT, Booster CC BY-NC 4.0. DINOv3 weights: DINOv3 License.
