# Reproducing the completed QAAP experiment

## Data and model availability

The four internal labels are `COVID`, `Normal`, `Lung_Opacity` and `Viral Pneumonia`. The source is the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). External data are the Shenzhen and Montgomery collections described by [Jaeger et al.](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20). Use the original metadata alongside the image collections; the external loader validates study-ID joins and does not invent demographic attributes.

Dataset versions, file paths and content affect the experiment signature. Attach the same inputs and preserve their names and folder layouts when restoring a run. A mismatch stops execution. Do not bypass this guard to reuse weights with a different cohort. Dataset and model redistribution permissions are separate from repository availability.

The notebook downloads a [GenSeg](https://github.com/importZL/GenSeg) frozen JSRT U-Net checkpoint and timm pretrained classification weights. Internet access is needed for initial retrieval. Provenance hashes are written by the implementation. The source images and the 3.94 GiB `QAAP_complete_run.zip` are not in this Git repository. The maintainer must supply a reviewer-accessible weights URL separately; none is fabricated here.

The maintainer's [Kaggle notebook reference](https://www.kaggle.com/code/rs1911/qaap-final) is distinct from a model dataset. Its editing URL must not be treated as a public weights download; public access could not be verified during this documentation update.

## Cohort and fitting boundaries

| Partition | COVID | Normal | Lung opacity | Viral pneumonia | Total |
|---|---:|---:|---:|---:|---:|
| Training | 2,495 | 7,134 | 4,208 | 937 | 14,774 |
| Tuning | 356 | 1,019 | 601 | 134 | 2,110 |
| Calibration | 357 | 1,019 | 601 | 134 | 2,111 |
| Test | 357 | 1,019 | 602 | 133 | 2,111 |

The 21,106 images are unique by decoded pixel hash. Stratified split seeds are 42, 43 and 44. The training folds use seed 42. Image-level independence does not establish patient-level independence.

Each fold's quality scales and SSL encoder use its training images only. Three base classifiers produce four-class probabilities for held-out images; 12 concatenated OOF features train logistic regression (`C=1`, `max_iter=2000`). Tuning scores are diagnostics, not best-epoch selection. Temperature scaling fits calibration NLL. Test labels do not fit either the stacker or temperatures.

## Environment

The recorded run used Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, timm 1.0.15 and Kornia 0.7.4. Two Tesla T4s were visible; models used one GPU. The execution policy used zero loader workers, no pinned loader memory and cuDNN autotuning off. See [environment.json](../results/environment.json) and [execution_policy.json](../results/execution_policy.json).

The notebook installation cell pins timm, Kornia, gdown and grad-cam. `requirements.txt` lists the remaining imports, but it is not a complete lock file: versions of NumPy, pandas, SciPy, scikit-learn, torchvision and other runtime packages were not all captured. Use a compatible CUDA-enabled PyTorch/torchvision pair provided by Kaggle. Installing CPU-only PyTorch does not reproduce the GPU environment. A local install alone will not recreate `/kaggle/input` paths.

Seeds are set, but deterministic CUDA execution is not enforced. New training need not reproduce the saved weights or metrics bit for bit. In particular, single-GPU BatchNorm sees global batch 32 rather than the per-device halves of earlier DataParallel experiments.

## Fresh experiment

1. Import `QAAP_Final.ipynb` into Kaggle and enable GPU and Internet.
2. Attach the three source collections and their original metadata.
3. Run Dependencies. Leave optional completed-run restoration disabled.
4. Run configuration and setup sections 2–7. Section 4 creates lung crops; it can take substantial time.
5. Run Backup helpers and Preparation backup. Save all parts and the manifest locally.
6. Run one fold cell at a time, from fold 1 to fold 5. Each produces a backup. There are no temporary `parallel()` overrides to paste.
7. Run final model cells in order: `cn`, `eff`, `dense`, `vit`, `cn_vanilla`, `cn_single`, `cn_noqaap`. Final SSL is reused after its first fit. Download each backup.
8. Run evaluation setup, then sections 9–13 for calibration, statistics, external transfer, Grad-CAM and latency. Download the newest reports ZIP and final evaluation backup.
9. Download the executed notebook separately. Verify local downloads before stopping the session; printed links and saved cell output are not trained-weight backups.

Global batch size is 32; image size is 224. All classifiers train for five epochs and SSL for one epoch with a 65,536-entry queue. AdamW uses weight decay 0.0001, supervised learning rate 0.0001 and SSL learning rate 0.0003, with epoch-wise cosine scheduling. The final epoch is saved; there is no early stopping or best-validation selection. Mid-epoch checkpoints are written every 100 batches, resource samples every 50 batches.

## Restoring completed weights

Attach the maintainer's `QAAP_complete_run.zip` as a Kaggle input. Set `RESTORE_COMPLETED_RUN = True` in the optional restore cell before configuration. It supports either the ZIP or the extracted experiment folder and refuses to overwrite an existing working experiment. It verifies 28 neural weight files. The main implementation subsequently checks the experiment signature. Image caches were deliberately excluded and may need rebuilding.

Completed weight files are reused by the training functions. To avoid running backup cells again when only evaluating, run setup through section 7 and Backup helpers, then jump to evaluation setup and sections 9–13. Do not run the fold/final-training cells in that evaluation-only workflow. Archived notebook tables do not require restoration or execution to view.

## Split backups and session recovery

Each stage download consists of `QAAP_backup_<stage>.zip.partNNN` files plus its `.manifest.json`. Parts are at most 200 MiB and are not independently readable ZIPs. Download all parts; preserve their exact names. Browser-renamed duplicates such as `(1)` do not replace the manifest's named original.

```bash
python tools/verify_stage_backups.py /path/to/downloads
python tools/verify_stage_backups.py /path/to/downloads --join
```

Verification checks exact sizes and SHA-256 digests. `--join` also validates ZIP CRCs and does not load model pickles. Joined stages share the `qaap_v3_optimized` root. Restore preparation, folds and final-model stages in chronological order, followed by evaluation. Keep newer weights; do not overwrite them with an old partial checkpoint. Caches are regenerable. An interrupted stage's `_latest.pth` needs a separate partial-checkpoint backup because completed-stage backups intentionally omit unfinished checkpoints.

## Evidence and reproducibility limits

The code snapshot is consolidated; its code cells have empty outputs. The appended tables/figures and `results/` exports come from the verified completed run. [The report](QAAP_final_research_report.pdf) additionally calculates exploratory component bootstrap intervals and reconstructs pre-calibration stacking scores from saved probabilities; those additional analyses are not all separate output cells in the notebook.

The earlier fast baseline's matching test predictions were not supplied, so the original two-percentage-point acceptance comparison remains pending. The change from the first optimized run's 95.6013% macro-F1 to 95.5151% in the final run is not that baseline comparison. Logged epoch segments are not a complete wall-clock runtime. No new GPU training was performed when adding repository documentation.
