# Kasra: ConvNeXt-Tiny baseline (`kasra/convnext-baseline`)

**Notebook:** `01_convnext_baseline.ipynb`
**Runs on:** a Kaggle notebook with a GPU (Internet on, competition data attached)

## What it does
- Stratified 5-fold CV on the training set
- Model: ConvNeXt-Tiny, pretrained on ImageNet-22k, input 256 px
- Augmentation: rotations and flips, plus color jitter, blur, noise and random erasing for robustness to corrupted images
- Loss: cross-entropy with label smoothing 0.1 and sqrt-inverse-frequency class weights
- Training: AdamW, one-cycle cosine LR, AMP, EMA weights, early stopping on weighted F1
- Prediction: 8-view TTA and a fold ensemble

## Outputs (in `/kaggle/working`)
| File | Use |
|---|---|
| `submission.csv` | upload to Kaggle |
| `test_probs.npy` | class probabilities on test, for the team ensemble |
| `oof_probs.npy` | out-of-fold probabilities, for local CV and ensembling |
| `suspicious_labels.csv` | likely mislabeled training images, for the cleaning step |
| `train_anomaly_ranking.csv` | images with unusual colors/statistics, likely corrupted |
| `cv_score.json` | local CV score |

## Results log
| Date | Folds | Local CV wF1 | Public LB | Notes |
|---|---|---|---|---|
|  | [0] |  |  | first baseline |
