# Improved U-Net for Prostate MRI Segmentation

**COMP3710 Pattern Recognition, UQ, Semester 2 2026 — hetmehta0**

Compare a standard 2D U-Net baseline with a residual, deeply supervised 2D
Improved U-Net on HipMRI prostate segmentation under the **normal difficulty**
option. The intended output is a proposed anatomical mask for expert review.
The research question is whether the improved model reduces apex/base and
adjacent-organ boundary errors enough to justify its extra computation.
**Held-out prostate Dice >= 0.75 is a target, not an achieved result.**

## Status through Day 4

The Day 1 structure and initial commit were verified and preserved. Days 3–4
software is implemented and tested locally. Day 2 has a feasibility review and
audit tooling; **the actual HipMRI inspection is blocked**. The user has no
Rangpur access or local dataset copy. No actual HipMRI model has been trained
or evaluated. GPU execution has not been tested.

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

**Prepared 20 October 2026. Status: software evidence only; tutor check-off pending.**

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

**Preliminary evidence.** The local checkout contained the Day 1 commit and
empty Python placeholders. The Rangpur directory is absent locally; the user
confirmed no login or dataset copy. Real patient identities, label meanings,
slice counts, balance and geometry remain unverified. Synthetic Nifti tests
exercise 10 invented patients and 20 slices, partitioned 8/1/1 patients.
Both models completed two CPU epochs, checkpoint reload, native-grid evaluation
and two prediction exports each. At width 2 and 32x32 inputs, baseline/improved
have 30,537/44,025 parameters; observed mean warmed forward times over two test
slices were 0.531/1.486 ms. This tiny synthetic timing cannot estimate full
HipMRI training or GPU performance.

**Risks, budget and fallback.** Main risks are unavailable data, incorrect
patient/label mapping, small-cohort variance, resize-related boundary loss and
2D continuity failures. After access: inventory all files, verify ontology and
patient mapping with source documentation, visually audit pairs, lock the split,
then run a small real-data optimization check. Plan one 50-epoch run per model
at width 16, 256x256, batch 8; measure a real epoch before reserving GPU hours.
Use the same reduced batch size for both if memory is insufficient. Retain the
baseline if the improved model lacks meaningful boundary benefit; discuss any
difficulty change with the tutor. No superiority or deployment recommendation
is justified yet. Present this review at the compulsory practical consultation.

## Installation and reproducibility

Run from this folder. Tested on Python **3.12.2**, macOS arm64, CPU, one thread.
Exact direct package versions are in `requirements.txt`; the full tested
environment is in `requirements-lock.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python smoke.py --out runs/synthetic-smoke
```

Use a new output directory on each run: training and prediction refuse to
overwrite an existing directory. On Rangpur, select a site-compatible CUDA
PyTorch build and record the actual environment with `python -m pip freeze`.
The macOS lock is evidence of local validation, not a CUDA environment guarantee.
CPU is the default; CUDA/MPS must be explicitly requested. Deterministic
operations are requested; unsupported GPU deterministic operations may fail,
and exact numerical equivalence across devices is not promised.

`smoke.py` generates synthetic Nifti fixtures, trains both models for two epochs,
reloads their best checkpoints, evaluates synthetic test patients and exports
panels and masks. Its tiny width-2/32x32 configuration is **not** the planned
real-data experiment. Its logs explicitly identify synthetic provenance.

## Real-data preparation once access is available

Keep data, patient manifests, audits and checkpoints outside Git, or under the
ignored `private/`, `data/` and `runs/` directories.

```bash
python dataset.py inventory \
  --root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
  --out private/inventory.json
```

The inventory records actual filenames, shapes, spacing, dtype, nonfinite
counts, intensity range and unique values for files with <=32 distinct values.
It does **not** assign meanings to raw integers or guess patient identities.
Confirm those against dataset documentation/tutor guidance.

Create `private/pairs.csv` with columns:

```csv
patient_id,case_id,image,mask
```

Supply one row per aligned 2D image/mask pair; `case_id` uniquely identifies a
slice, while `patient_id` groups **every slice and repeated scan of that person**.
Relative image/mask paths resolve beside the CSV. Keep these identifiers private.
Review the correspondence visually, not just lexicographically.

Create `private/classes.json` as an ordered list of objects with integer `value`
and string `name` fields. The first entry must be `background`, and one entry
must be `prostate`; include every observed class. **No actual HipMRI raw-label
mapping is supplied because it has not been verified.** The fixture vocabulary
0/2/5 is invented and must not be reused as HipMRI metadata.

```bash
python dataset.py prepare --pairs private/pairs.csv \
  --classes private/classes.json --out private/manifest.json --seed 3710
```

The audit checks every pair, raw label, shape and affine; rejects reused paths
(including symlinks), identical image pixels assigned to different patients,
and partitions with no prostate pixels. It records raw pixel counts and split
sizes. Split fractions request 70/15/15; validation/test patient counts are
floored with a minimum of one each, so actual fractions depend on cohort size.
Small cohorts may need tutor-approved patient stratification before locking the
manifest. Never choose a split by its eventual test performance.

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
python train.py fit --manifest private/manifest.json --model baseline \
  --out runs/baseline --epochs 50 --size 256 256 --batch-size 8 \
  --base-channels 16 --seed 3710 --device cuda
python train.py fit --manifest private/manifest.json --model improved \
  --out runs/improved --epochs 50 --size 256 256 --batch-size 8 \
  --base-channels 16 --seed 3710 --device cuda
```

Each run saves configuration/environment, epoch losses, validation scores,
learning curves, parameter count, elapsed training/validation time and best
weights. CUDA reports peak allocated bytes; CPU/MPS report CUDA memory as
`null`, never zero. Optimizer resume is not implemented.

After freezing model choices, evaluate the test set once for final reporting:

```bash
python train.py evaluate --manifest private/manifest.json \
  --checkpoint runs/baseline/best.pt --split test \
  --out runs/baseline/test_metrics.json --device cuda
python train.py evaluate --manifest private/manifest.json \
  --checkpoint runs/improved/best.pt --split test \
  --out runs/improved/test_metrics.json --device cuda
python predict.py --manifest private/manifest.json \
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

The end-to-end CLI smoke completed on 20 October 2026: 10 **synthetic** patients,
20 slices of shape 33x35x1, split 8/1/1 patients, training size 32x32, base width
2, three invented labels, seed 3710, batch 4, two epochs, CPU one thread.
Both models reloaded checkpoints and exported two test masks/panels each.
Separate evaluation and prediction metrics agreed exactly.

| Synthetic smoke measurement | Baseline | Improved |
|---|---:|---:|
| Trainable parameters, width 2, three classes | 30,537 | 44,025 |
| Two-epoch training + validation time (seconds) | 0.098 | 0.129 |
| Warmed model-only forward, mean of two slices (ms) | 0.531 | 1.486 |
| Peak CUDA allocated memory | Not measured (CPU) | Not measured (CPU) |
| Synthetic test prostate Dice (not HipMRI performance) | 0.154952 | 0.217105 |

These numbers demonstrate execution only. Two cases, toy shapes and tiny networks
cannot support a speed comparison or a clinical/model-quality conclusion.
Forward timing excludes I/O, preprocessing and exports. The width-16 models with
three classes contain 1,940,851 and 2,774,249 parameters, respectively; actual
counts depend on the verified HipMRI class vocabulary.

Tests cover patient isolation, deterministic partitions, changed files,
noncontiguous labels, missing/unknown classes, geometry errors, nonfinite
data, real-volume rejection, paired augmentation, all-head gradients, hand-computed
metrics, checkpoint identity, loss reduction and exported Nifti geometry.
See `DEVELOPMENT.md` for recorded test runs.

## Real experiments and error analysis — pending

| Required real-data evidence | Status |
|---|---|
| Observed patient/slice counts, identities, labels and balance | Blocked by access |
| Baseline vs improved held-out per-class Dice/IoU | Not run |
| Rangpur training time, peak GPU memory and inference time | Not measured |
| 3–5 actual apex/base/boundary failure autopsies | Not possible without data |
| Clinical boundary utility/resource recommendation | Deferred until evidence exists |
| Tutor feasibility check-off | User must attend |

Once data is available, profile one real epoch, verify optimization on a small
training-only subset, run the controlled comparison, then inspect failure cases.
Retain test independence during tuning. Do not infer clinical benefit from the
synthetic figures. The final submission additionally needs completed real
experiments, report analysis, a user-authorized upstream PR and README PDF for
Turnitin; these have not been submitted.

## Artificial Intelligence Usage Disclosure

OpenAI **Codex** assisted with specification interpretation, architecture design,
the Nifti loader and split audit, model/loss/metric implementation, training and
prediction scripts, synthetic fixtures, tests and documentation. The earlier
ChatGPT conversation was reviewed for context; its incomplete responses were
not used as assessment authority. No prior student solution was copied.

Verification performed by the assistant includes automated negative tests for
leakage and invalid geometry/labels, hand-derived metric examples, gradient
checks, repeated-batch optimization, CPU end-to-end runs, checkpoint reload
consistency and native-label/geometry export tests. These are **assistant-run
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
