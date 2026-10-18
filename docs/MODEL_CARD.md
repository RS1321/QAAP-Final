# QAAP experiment model card

## Intended scope

This repository records one research experiment classifying chest radiographs into COVID, Normal, Lung Opacity and Viral Pneumonia. The ensemble uses ConvNeXt-small, EfficientNetV2-S and DenseNet-201 with a logistic stacker. DeiT-small and ConvNeXt ablations are comparators. Clinical validity, prospective screening performance and patient-level independence have not been established.

## Training and preprocessing

Native-depth image normalization, frozen GenSeg JSRT U-Net lung cropping and quality-feature extraction precede learned soft-CLAHE/bilateral/unsharp enhancement. Invalid masks fall back to the full image. ConvNeXt models receive one epoch of domain SSL with a MoCo-style negative queue; this is not canonical MoCo v3. Supervised models train for five epochs under the same fixed budget. All fitted fold-specific components exclude each fold's held-out images.

## Evaluation

- Internal test: 2,111 images; stacking accuracy 95.17%, macro-F1 95.52%, AUC 0.9933, ECE 0.0133.
- External transfer: 800 Shenzhen/Montgomery images; score `1 - P(Normal)`, threshold 0.5; stacking macro-F1 75.29%, sensitivity 61.42%, specificity 89.90%. TB is not a training class.
- Batch-one end-to-end latency: mean 107.62 ms and p95 112.53 ms; the 100 ms p95 target failed.
- Uncertainty: 1,000 paired stratified image bootstraps, plus exact McNemar correctness tests and Holm correction for eight stacking comparisons. They do not measure repeated-training variability.

## Limitations supported by this run

1. Single-scale ConvNeXt outperformed the full multi-scale version. QAAP's macro-F1 effect was small and inconclusive. Added components must not all be described as beneficial.
2. Stacking did not establish superiority over soft voting or DeiT-small in the reported paired comparisons. Temperature scaling slightly worsened stacking's internal test ECE.
3. Splits are image-level. Patient grouping, full near-duplicate detection and acquisition-confound control are not established.
4. Frozen masks sometimes have poor Dice. Eight selected Grad-CAM examples had less than 50% heat mass inside the predicted lung masks; this is neither ensemble localisation validation nor proof of a causal failure mechanism.
5. External subgroup findings vary by site and have small denominators, no subgroup confidence intervals and no case-mix adjustment. Manufacturer values are unavailable for all 800 records.
6. External TB transfer has different labels and score definitions from internal four-class testing. Its scores must not be presented as validated TB diagnosis or calibrated external risk.
7. One trained realization per configuration and five supervised epochs do not establish convergence or ranking stability across seeds. A full environment lock and the original-baseline acceptance comparison are unavailable.

## Availability and permissions

Code and aggregate results are in this repository. Images, full model weights and per-image probability archives are separate artifacts. No public model-artifact URL has been supplied here. The pretrained models and original datasets retain their own licences and terms. No repository-wide software licence is currently selected.
