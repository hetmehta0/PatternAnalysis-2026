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

## Day 2 access evidence

The specified Rangpur path
`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`
does not exist on this Mac. No Rangpur SSH configuration was found. The user
confirmed on 20 October that they have neither login access nor a local copy.
No remote dataset inspection or remote authentication was performed.
Actual patient count, slice count, shape distribution, raw label IDs, identity
mapping, alignment and class balance remain **unverified**.

Appendix A describes 38 patients and 211 3D volumes; these are specification
figures, not observations of the 2D directory and not evidence of accessible data.
Synthetic fixtures will verify software behavior only.

## Integrity

OpenAI Codex assisted with this implementation and documentation. Commits record
actual development milestones on 20 October; they do not imply work on separate
days or independently authored student work. The student must inspect, understand,
verify and defend this code. No real-data performance claims or tutor check-off
completion are asserted. No pull request is authorized at this stage.
