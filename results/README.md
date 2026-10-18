# Verified aggregate results

These files are exports from the completed staged experiment, not newly generated training results. They agree with the attached research report. Fractional metrics are on a 0–1 scale unless their column name states milliseconds. Multiply accuracy and F1 by 100 to report percentages; multiply paired F1 differences by 100 for percentage points.

| File | Contents |
|---|---|
| `test_results.csv` | Calibrated internal test metrics, 2,111 images, nine models/ensembles |
| `bootstrap_ci.csv` | Accuracy, macro-F1 and AUC percentile intervals from 1,000 shared stratified resamples |
| `paired_comparisons.csv` | Stacking-minus-baseline F1 differences and exact paired correctness tests with Holm adjustment |
| `tb_generalization.csv` | Pooled external normal-versus-abnormal transfer on 800 TB-labelled/Normal images |
| `tb_demographic_metrics.csv` | Site, sex, age and intersectional subgroup counts and metrics |
| `tb_demographic_gaps.csv` | AUC max–min gaps among eligible known demographic groups |
| `gradcam_attention.csv` | Lung heat fractions for eight selected DenseNet illustrations |
| `latency.csv` | Mean, median and p95 milliseconds by stage, batch one, ten warm-ups and 50 images |
| `measured_ablation_results.csv` | Measured variant results; not a fabricated expected-outcomes table |
| `preprocessing_benchmark.json` | Synthetic full QAAP forward/backward benchmark; not end-to-end speedup |
| `environment.json`, `execution_policy.json` | Recorded software/GPU environment and resource policy |
| `manufacturer_status.json` | Zero known manufacturer values among 800 external records |
| `settings.json` | Actual experiment settings, distinct from loader/resource overrides |
| `figures/` | Original internal confusion matrix and measured-ablation figure |

The full artifact also contains per-image predictions, the manifest, calibration parameters, segmentation provenance and neural weights; they are not included in this small aggregate export. Selected figure provenance and further analyses are described in [the report](../docs/QAAP_final_research_report.pdf). No confidence interval here accounts for repeated training or verified patient clustering. McNemar p-values test correctness, not macro-F1. Undefined subgroup values must not be interpreted as zero performance.
