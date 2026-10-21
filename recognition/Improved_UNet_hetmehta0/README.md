# Improved U-Net for Prostate MRI Segmentation

**COMP3710 Pattern Recognition, UQ, Semester 2 2026 — hetmehta0**

Compare a standard 2D U-Net baseline with a residual, deeply supervised 2D
Improved U-Net on HipMRI prostate segmentation under the **normal difficulty**
option. The intended output is a proposed anatomical mask for expert review.
The research question is whether the improved model reduces apex/base and
adjacent-organ boundary errors enough to justify its extra computation.
**Held-out prostate Dice >= 0.75 is a target, not an achieved result.**

## Status through Day 4

The Day 1 structure and initial commit were verified and preserved. Days 2–4
are now implemented and tested with the downloaded HipMRI 2D data. All 12,660
image/mask pairs were audited, the supplied patient-disjoint partitions were
preserved, and both architectures completed a small real-data pipeline smoke.
Full representative training and GPU execution have not yet been performed.

The authority used was the supplied `COMP3710_Report_2026_draft_v2.3.pdf`
(its first page labels itself “Version 2.2 Final Assessment Specification”).
Relevant sections: 2.2, 3, 4.1, 5.3, 6.1 and Appendices A–B.
The tutor feasibility consultation is a **compulsory pass hurdle and remains
outstanding**. Verify its date in the course profile. No PR has been opened.

## How the models work

Both models contract spatial resolution four times, then recover resolution
using encoder-to-decoder skip concatenations. The baseline uses double 3x3
convolutions, ReLU, max pooling and transposed-convolution upsampling, following
[Ronneberger et al.](https://arxiv.org/abs/1505.04597). Same padding and a
configurable starting width are deliberate implementation choices.

The Improved U-Net is a **2D adaptation** inspired by the architecture in
[Isensee et al., Section 2.1](https://arxiv.org/abs/1802.10508): residual context
blocks, instance normalization, leaky ReLU (0.01), dropout (0.3), stride-2
convolutions, and localization blocks. Three decoder heads produce logits that
are upsampled and summed; the segmentation loss sends gradients into all heads.
Adaptations include 2D kernels, one input channel, configurable widths and
interpolation directly to skip dimensions. There is no test-time ensemble.
This is not a claim of exact reproduction or of tutor approval.

```mermaid
flowchart LR
  A[Reviewed patient/pair CSV + label dictionary] --> B[Geometry and label audit]
  B --> C[Seeded patient partitions + file hashes]
  C --> D[Train: normalization, resize, paired flips]
  D --> E[Baseline or Improved U-Net]
  E --> F[Validation prostate Dice selects checkpoint]
  F --> G[Explicit held-out evaluation]
  G --> H[Native-grid labels, Dice/IoU, FP/FN panels]
```

Neither model adds true through-plane 3D context. Therefore continuity
improvements must be demonstrated rather than assumed.

## One-page feasibility review

**Updated 21 October 2026. Status: real-data feasibility evidence; tutor check-off pending.**

**User need and scope.** An imaging specialist needs a prototype that drafts
prostate contours for review. Compare normal-difficulty 2D Improved U-Net with
standard 2D U-Net on the same held-out HipMRI patients. Investigate whether
residual multi-scale features reduce difficult boundary errors, despite the
absence of 3D continuity. This prototype is not approved for clinical use.

**Testable acceptance criteria (proposed).** (1) Zero patient overlap and no
reused files across partitions; reject misaligned pairs and unknown labels.
(2) Held-out patient-macro prostate Dice >= 0.75, with pooled and per-class
Dice/IoU also reported. (3) Review 3–5 real boundary failures, including apex/base
and adjacent-organ leakage, using input/truth/prediction/FP-FN panels.
(4) On the allocated GPU, target peak allocated memory <= 8 GiB and warmed
forward latency <= 1 second per 256x256 slice at batch 1. These are proposed
resource limits, not measured Rangpur results. (5) Compare both models using
the identical split, preprocessing, seed and training budget.

**Model choice and course concepts.** Use 2D Nifti slices, a verified fixed
multi-class vocabulary, per-slice normalization and patient-level partitions.
Convolutions, pooling/striding, skip connections, residual learning, dropout
and gradient supervision connect course concepts to the architecture. The
improved model is a 2D adaptation of Isensee et al.; the baseline follows
Ronneberger et al. Both use cross-entropy plus foreground soft Dice.

**Preliminary evidence.** The downloaded data contains 12,660 aligned slice
pairs from 38 patients. The supplied train/validation/test partitions contain
31/4/3 patients and 11,460/660/540 slices, with no patient overlap. Every pair
passed shape, affine and finite-value checks. Labels 0–5 match the published
order: background, bladder, body, bone, rectum and prostate. Prostate occupies
1,766,549 of 415,088,640 pixels (0.426%), confirming severe imbalance. Most
slices are 256x128; 60 training slices are 256x144. Visual checks of positive
examples from every split showed the label-5 contour aligned with the prostate.
Both models also completed a one-epoch CPU pipeline check on 114 real slices,
including checkpoint reload, native-grid evaluation and prediction export.

**Risks, budget and fallback.** Main risks are small-cohort variance,
resize-related boundary loss, extreme class imbalance and 2D continuity
failures. The next experiment is a short representative optimization run on the
full training partition, followed by one 50-epoch run per model
at width 16, 256x256, batch 8; measure a real epoch before reserving GPU hours.
Use the same reduced batch size for both if memory is insufficient. Retain the
baseline if the improved model lacks meaningful boundary benefit; discuss any
difficulty change with the tutor. No superiority or deployment recommendation
is justified yet. Present this review at the compulsory practical consultation.

## Installation and reproducibility

Run from this folder. Tested on Python **3.11.8 and 3.12.2**, macOS arm64, CPU,
one thread. Exact direct package versions are in `requirements.txt`; a full
captured test environment is in `requirements-lock.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python dataset.py prepare-hipmri --root /path/to/HipMRI-2D \
  --out private/hipmri-manifest.json
python smoke.py --manifest private/hipmri-manifest.json \
  --out runs/real-smoke
```

Use a new output directory on each run: training and prediction refuse to
overwrite an existing directory. On Rangpur, select a site-compatible CUDA
PyTorch build and record the actual environment with `python -m pip freeze`.
The macOS lock is evidence of local validation, not a CUDA environment guarantee.
CPU is the default; CUDA/MPS must be explicitly requested. Deterministic
operations are requested; unsupported GPU deterministic operations may fail,
and exact numerical equivalence across devices is not promised.

`smoke.py` deterministically selects two prostate-positive and one negative real
slice per patient, trains both models for one epoch, reloads checkpoints,
evaluates and exports predictions. This deliberately balanced subset is only an
execution check and cannot estimate performance on the natural distribution.
Synthetic Nifti remains confined to automated unit-test fixtures.

## Real-data preparation

Keep data, patient manifests, audits and checkpoints outside Git, or under the
ignored `private/`, `data/` and `runs/` directories.

```bash
python dataset.py prepare-hipmri --root /path/to/HipMRI-2D \
  --out private/hipmri-manifest.json --seed 3710
```

The indexer matches `case_...` images to `seg_...` masks by their shared
patient/week/slice key. It preserves the supplied folders as train, validation
and test because the observed patient IDs are disjoint. The audit checks every
pair, raw label, shape and affine; rejects reused paths, identical image pixels
assigned to different patients, and partitions with no prostate pixels. Label
mapping follows Dai et al. (2021): released Nifti values 0–5 correspond to
background, bladder, body, bone, rectum and prostate.

The manifest includes hashes of all files and its own contents. Train/evaluate/
predict verify it before use. Checkpoints bind to that exact manifest and class
order. Hashes detect input changes but cannot prove that manually supplied patient
identities or anatomical label meanings are correct. Moving data requires
regenerating the manifest and a new run.

## Preprocessing, optimization and metrics

- Accept only `(H,W)` and `(H,W,1)` Nifti, with matching shapes and affines.
  Reject conflicting coded qform/sform, nonfinite pixels, fractional masks and
  true volumes. Geometry checks cannot detect a semantically wrong but geometrically
  identical pair; manual inspection remains necessary.
- Normalize each original slice by its own mean and standard deviation; constant
  slices become zero. No statistics are fitted on other patients.
- Resize both arrays to the same target shape: bilinear image interpolation and
  nearest-neighbour mask interpolation. Default 256x256 is a provisional choice;
  aspect distortion and small-structure loss must be checked on real data.
  Do not resample an incorrectly aligned pair merely to make it pass.
- Encode integer labels into a fixed index dictionary and one-hot channels,
  including classes absent from a particular slice. Apply paired axis flips only
  during training; flips are seeded per case/epoch independently of model dropout.
- Optimize `0.5 * CE + 0.5 * (1 - foreground soft Dice)` with Adam
  (initial learning rate 0.001, weight decay 0.00001). Soft Dice uses
  `(2*sum(p*y)+1e-6)/(sum(p)+sum(y)+1e-6)` over batch/spatial axes.
  The background is excluded from the Dice loss, but included in CE.
- Select the best checkpoint using **validation patient-macro prostate Dice**;
  ties retain the earlier epoch. `fit` does not evaluate test metrics.
- Evaluate by resizing logits back to original resolution **before argmax**.
  Compare with untouched original masks. Dice is `2TP/(2TP+FP+FN)` and IoU is
  `TP/(TP+FP+FN)`. Report pooled counts, per-patient scores and patient-macro scores.
  Empty prediction and truth produce `null`, excluded from macro averages;
  a false positive on absent ground truth scores zero.
- Patient aggregation combines all supplied slices/scans by patient ID. It is
  not a spacing-weighted reconstructed 3D volume metric. Original affines, shape
  and raw label IDs are retained in exported Nifti masks.

## Training, held-out evaluation and prediction

Use identical settings and the **same manifest** for both architectures:

```bash
python train.py fit --manifest private/hipmri-manifest.json --model baseline \
  --out runs/baseline --epochs 50 --size 256 256 --batch-size 8 \
  --base-channels 16 --seed 3710 --device cuda
python train.py fit --manifest private/hipmri-manifest.json --model improved \
  --out runs/improved --epochs 50 --size 256 256 --batch-size 8 \
  --base-channels 16 --seed 3710 --device cuda
```

Each run saves configuration/environment, epoch losses, validation scores,
learning curves, parameter count, elapsed training/validation time and best
weights. CUDA reports peak allocated bytes; CPU/MPS report CUDA memory as
`null`, never zero. Optimizer resume is not implemented.

After freezing model choices, evaluate the test set once for final reporting:

```bash
python train.py evaluate --manifest private/hipmri-manifest.json \
  --checkpoint runs/baseline/best.pt --split test \
  --out runs/baseline/test_metrics.json --device cuda
python train.py evaluate --manifest private/hipmri-manifest.json \
  --checkpoint runs/improved/best.pt --split test \
  --out runs/improved/test_metrics.json --device cuda
python predict.py --manifest private/hipmri-manifest.json \
  --checkpoint runs/improved/best.pt --split test \
  --out runs/improved/predictions --max-panels 5 --device cuda
```

Repeat `predict.py` for the baseline to compare identical case IDs. It exports
all selected masks, metrics and up to five panels by default. For a complete
visual audit increase `--max-panels`. Panels show all-class truth/prediction and
prostate FP (red)/FN (blue). The first five cases are not automatically the most
informative failures: select 3–5 actual failure cases after review, using verified
slice-position metadata to identify base/apex. Display coordinates are array
coordinates, not a claimed radiological viewing convention.

## Observed software verification

The end-to-end real-data CLI smoke completed on 21 October 2026. It used 114
HipMRI slices: two prostate-positive and one negative slice per patient,
partitioned as 93/12/9 train/validation/test slices across the original 31/4/3
patients. Inputs were resized to 64x64, base width was 2, batch size was 8 and
each model ran for one CPU epoch. Both models reloaded their checkpoint and
exported nine test masks/panels. Evaluation and prediction metrics agreed.

| Real-data pipeline smoke measurement | Baseline | Improved |
|---|---:|---:|
| Trainable parameters, width 2, six classes | 30,546 | 44,076 |
| One-epoch training + validation time (seconds) | 0.770 | 1.032 |
| Warmed model-only forward, mean of nine slices (ms) | 1.404 | 3.785 |
| Peak CUDA allocated memory | Not measured (CPU) | Not measured (CPU) |
| Balanced-subset test prostate Dice (not a benchmark) | 0.002154 | 0.005610 |

These numbers demonstrate execution only. One epoch, tiny networks and a
deliberately balanced subset cannot support a clinical/model-quality conclusion.
Forward timing excludes I/O, preprocessing and exports. The width-16 models with
six classes contain 1,940,902 and 2,774,594 parameters, respectively.

Tests cover patient isolation, deterministic partitions, changed files,
noncontiguous labels, missing/unknown classes, geometry errors, nonfinite
data, real-volume rejection, paired augmentation, all-head gradients, hand-computed
metrics, checkpoint identity, loss reduction and exported Nifti geometry.
See `DEVELOPMENT.md` for recorded test runs.

## Real experiments and error analysis — pending

| Required real-data evidence | Status |
|---|---|
| Observed patient/slice counts, identities, labels and balance | Complete |
| Full image/mask shape, affine and finite-value audit | Complete |
| Baseline vs improved held-out per-class Dice/IoU | Not run |
| Rangpur training time, peak GPU memory and inference time | Not measured |
| 3–5 actual apex/base/boundary failure autopsies | Not yet completed |
| Clinical boundary utility/resource recommendation | Deferred until evidence exists |
| Tutor feasibility check-off | User must attend |

Next, profile one full-partition epoch, verify optimization on a small
training-only subset, run the controlled comparison, then inspect failure cases.
Retain test independence during tuning. Do not infer clinical benefit from the
pipeline-smoke figures. The final submission additionally needs completed real
experiments, report analysis, a user-authorized upstream PR and README PDF for
Turnitin; these have not been submitted.

## Artificial Intelligence Usage Disclosure

OpenAI **Codex** assisted with specification interpretation, architecture design,
the Nifti loader and split audit, model/loss/metric implementation, training and
prediction scripts, real-data indexing/smoke workflow, synthetic unit-test
fixtures, tests and documentation. The earlier
ChatGPT conversation was reviewed for context; its incomplete responses were
not used as assessment authority. No prior student solution was copied.

Verification performed by the assistant includes automated negative tests for
leakage and invalid geometry/labels, hand-derived metric examples, gradient
checks, repeated-batch optimization, CPU end-to-end runs, checkpoint reload
consistency, a complete real-data audit, visual alignment review, real-data
pipeline smoke, and native-label/geometry export tests. These are **assistant-run
checks**, not a claim that the student has independently audited the code.
The student must read, understand, audit and defend every line and every result,
verify actual metadata and real-data behavior, and update this disclosure with
their own verification before submission. Progressive commits were made during
one work session and do not imply separate days of independent student authorship.

## References

1. Supplied COMP3710 Report 2026 assessment specification, sections cited above.
2. Ronneberger, Fischer & Brox (2015), [U-Net](https://arxiv.org/abs/1505.04597).
3. Isensee et al. (2018), [Brain Tumor Segmentation and Radiomics Survival
   Prediction: Contribution to the BRATS 2017 Challenge](https://arxiv.org/abs/1802.10508).
4. Dai et al. (2021), [CAN3D: Fast 3D Medical Image Segmentation via Compact
   Context Aggregation](https://arxiv.org/abs/2109.05443).
5. Dowling & Greer (2021), [Labelled weekly MR images of the male pelvis](https://doi.org/10.25919/45t8-p065).
