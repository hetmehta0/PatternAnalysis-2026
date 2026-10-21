# COMP3710 development evidence

## Day 1 verification — 20 October 2026 (Australia/Brisbane)

- Existing clean branch: `topic-recognition`, fork `hetmehta0/PatternAnalysis-2026`;
  upstream `shakes76/PatternAnalysis-2026`.
- Preserved initial commit `f3c6563` (19 October): README, ignore rules and four
  empty Python placeholders. No prior model or loader implementation existed.
- The default Python 3.13.2 did not have PyTorch installed. Therefore the earlier
  environment-completion claim could not be verified in that interpreter.
  A separate Python 3.12 environment is being used for validation.
- Authority: supplied `COMP3710_Report_2026_draft_v2.3.pdf`; the document itself
  says "Version 2.2 Final Assessment Specification" and "(DRAFT)" on page 1.
  Consult the current course profile for deadlines; none is invented here.
- Relevant requirements: segmentation pp. 5–6, compulsory feasibility pp. 11–12,
  structure pp. 12–13, reporting p. 14, AI disclosure p. 15, data pp. 17–19.

## Methodology fixed before experiments

Compare a standard 2D U-Net with a 2D adaptation of Isensee et al.'s improved
U-Net: residual context blocks, instance normalization, leaky ReLU, spatial
dropout, learned downsampling and multi-scale segmentation heads. This is an
adaptation, not an exact reproduction of the original 3D BraTS model.
Both models use the same patient partitions, preprocessing, training budget,
loss, seed and validation selection rule. The baseline uses ordinary double
convolutions, max pooling and skip concatenations. Architecture acceptance
should be discussed with the tutor.

Use a fixed, documented class vocabulary including background and prostate,
rather than inferring channel numbers independently on each slice. Verify raw
label meanings from dataset documentation. Split unique patients before
augmentation; keep all scans/slices of each patient in one partition. Select
checkpoints by validation prostate Dice; reserve the test set for final reporting.

Clinical research question: does the improved model reduce apex/base and
adjacent-organ boundary failures enough to justify its measured resource cost?
Neither architecture adds true through-plane 3D context. Dice >= 0.75 is a
target, not an achieved result. Report per-class Dice/IoU, patient-level summaries,
resource measurements, and 3–5 real failure-case panels before making a recommendation.

## Day 2 initial access evidence — 20 October 2026

The specified Rangpur path
`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`
does not exist on this Mac. No Rangpur SSH configuration was found. The user
confirmed on 20 October that they have neither login access nor a local copy.
No remote dataset inspection or remote authentication was performed.
At that point, patient count, slice count, shape distribution, raw label IDs,
identity mapping, alignment and class balance remained **unverified**.

Appendix A describes 38 patients and 211 3D volumes; these were specification
figures rather than observations of the 2D directory.

## Day 2 access resolved — 21 October 2026

The user downloaded the official HipMRI 2D folders locally. The indexer found
12,660 paired Nifti slices from 38 patients and preserved the supplied
train/validation/test partitions: 31/4/3 patients and 11,460/660/540 slices.
No patient crossed a partition. Every pair passed shape, affine, finite-value
and known-label checks. There were 12,600 slices shaped 256x128 and 60 training
slices shaped 256x144, all with recorded in-plane spacing 1.0x1.0. Labels 0–5
were present; the published order maps them to background, bladder, body, bone,
rectum and prostate. Prostate occupied 1,766,549 of 415,088,640 pixels (0.426%).
Positive slices from all three partitions were manually inspected for anatomical
alignment. The complete ignored manifest fingerprint is
`7909783524416e0ce08197ca5f5b682e6cdd7677e95236ada7243b0b55cf84d7`.

## Integrity

OpenAI Codex assisted with this implementation and documentation. Commits record
actual development milestones on 20 October; they do not imply work on separate
days or independently authored student work. The student must inspect, understand,
verify and defend this code. No real-data performance claims or tutor check-off
completion are asserted. No pull request is authorized at this stage.

## Day 3 loader milestone

Implemented an audited, hash-sealed CSV-to-manifest workflow, seeded patient
partitioning, strict 2D Nifti geometry checks, per-slice normalization, fixed
one-hot channels and training-only paired flips. Initial loader suite:
**14 passed in 17.87s** on local CPU. Tests include leakage,
reused files, changed inputs, noncontiguous/unknown labels, affine/shape errors,
nonfinite data, fractional labels, volume rejection and constant normalization.

## Day 4 model and workflow milestone

Implemented the standard 2D U-Net baseline, a documented 2D adaptation of the
Isensee et al. improved U-Net, combined cross-entropy/foreground soft-Dice loss,
patient-level Dice/IoU aggregation, validation-only checkpoint selection,
explicit held-out evaluation, original-grid Nifti prediction export, and FP/FN
panels.

## Day 4 real-data rerun — 21 October 2026

The real-data smoke selected, for each patient, two prostate-positive slices and
one negative slice when available. This produced 114 slices across all 38
patients while retaining their official partitions (93/12/9 slices). Both
models completed one CPU epoch at 64x64 and width 2, reloaded the selected
checkpoint, evaluated on the original label grid and exported nine test masks.
Evaluation and prediction counts matched. Baseline/Improved parameter counts
were 30,546/44,076; training plus validation took 0.745/1.142 seconds and warmed
model-only forward latency was 1.343/3.434 ms per slice. The resulting prostate
Dice values (0.002154/0.005610) are deliberately recorded only as smoke outputs:
one epoch, tiny networks and a balanced subset cannot estimate model quality.
After the real-data indexer and supplied-split tests were added, the complete
suite passed: **25 passed in 6.66s** on Python 3.11.8, PyTorch 2.14.1, macOS
arm64 CPU with one thread.
