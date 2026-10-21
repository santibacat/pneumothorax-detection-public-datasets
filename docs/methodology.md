# Training of the multi-label EfficientNet-B5 multisource model

This document describes the procedure followed in the notebook [`notebooks/train/5.2 PTX ALL multilabel EffB5 train.ipynb`](../notebooks/train/) to train the multi-label chest radiograph model that uses all six development datasets (the *multisource* model).

It was reconstructed from the code, the execution counters, the saved outputs, the checkpoint CSV files and the helper functions of the `cxr` package. The notebook is not a clean linear execution: it contains resumed runs, alternative experiments and results from more than one session. For that reason the text distinguishes the **main flow** (the one that produced the released model), **later experimental branches** (discarded) and **older results that must not be mixed with it**. The log of the released model's run (Weights & Biases id `771kvn05`) reports 358,410 training images and 27,535 validation images, which confirms the main flow described below.

The six *singlesource* models were trained with the same recipe on each dataset alone (notebooks `5.3.x`).

## Training objective

A multi-label convolutional neural network was trained to predict three binary variables simultaneously:

1. `pneumothorax`: presence of pneumothorax;
2. `support_devices`: presence of support devices;
3. `no_finding`: absence of radiological findings.

Pneumothorax was the main task. The other two labels were added as auxiliary tasks.

## Execution environment

The documented run used:

- Python 3.10.14;
- TensorFlow 2.15.1;
- one NVIDIA GeForce RTX 2080 Ti GPU (about 10 GB of GPU memory available);
- Weights & Biases to log configurations, metrics and figures.

Training used RGB images and no mixed precision (`float16=False`).

## 1. Datasets

The development set combined six public datasets:

- NIH ChestX-ray14;
- CheXpert;
- MIMIC-CXR;
- PadChest;
- VinDr-CXR;
- SIIM-ACR Pneumothorax Segmentation.

Before filtering by projection, the loader gathered 904,454 images:

| Dataset | Images loaded |
|---|---:|
| MIMIC-CXR | 377,110 |
| CheXpert | 224,316 |
| PadChest | 160,861 |
| NIH | 112,120 |
| VinDr-CXR | 18,000 |
| SIIM-ACR | 12,047 |
| **Total** | **904,454** |

The loader reported 259,096 unique patients before the subsequent filters.

## 2. Cleaning and projection selection

Two lists of problematic files were imported: `bad_study_filepaths` and `bad_lat_filepaths` (`cxr/data_changes/data_changes.py`).

The projection of VinDr-CXR and SIIM-ACR was set manually to PA. For the remaining datasets the available metadata were used. Projections were grouped as follows:

- AP and PA: frontal (`lateral=0`);
- L, RL and LL: lateral (`lateral=1`);
- unknown or other projections: unclassified.

Only images with `lateral == 0`, i.e. AP or PA projections, were kept. The result, saved as `all_trainval_df_ckpt1.csv`, contains 687,864 frontal radiographs from 257,278 patients:

| Dataset | Frontal radiographs |
|---|---:|
| MIMIC-CXR | 243,334 |
| CheXpert | 191,730 |
| NIH | 112,120 |
| PadChest | 110,633 |
| VinDr-CXR | 18,000 |
| SIIM-ACR | 12,047 |
| **Total** | **687,864** |

There are no duplicated image paths in this checkpoint.

## 3. Label harmonization

Labels from the different datasets were harmonized with the following rules:

- only the value exactly equal to 1 was considered positive for pneumothorax;
- any other pneumothorax value, including 0, −1 (uncertain) and missing values, was converted to 0;
- missing `no_finding` values were replaced by 0;
- missing `support_devices` values were replaced by 0;
- `support_devices` values of −1 were also replaced by 0;
- when `pneumothorax=1`, `no_finding` was forced to 0 in the balanced training set.

After harmonization the complete frontal checkpoint had:

| Label | Negative | Positive |
|---|---:|---:|
| Pneumothorax | 650,630 | 37,234 |
| Support devices | 504,640 | 183,224 |
| No finding | 473,033 | 214,831 |

## 4. Training / validation split

A patient-level stratified split was performed independently within each dataset, using pneumothorax presence as the stratification variable.

The function used first grouped the images by `patient_uuid`, assigned to each patient the maximum of the pneumothorax label, and applied `StratifiedGroupKFold`. A validation fraction of 4% was configured (25 folds) and the first fold was taken. The default seed of this function was 42.

The result was saved as:

- `all_train_df_ckpt2.csv`: 660,329 images from 246,985 patients;
- `all_valid_df_ckpt2.csv`: 27,535 images from 10,293 patients.

No patient or image path overlap between training and validation was found. These assignments are published in [`splits/development_splits.csv.gz`](../splits/).

### Initial distribution

| Set | PTX negative | PTX positive | Total |
|---|---:|---:|---:|
| Training | 624,488 | 35,841 | 660,329 |
| Validation | 26,142 | 1,393 | 27,535 |

Pneumothorax prevalence was 5.43% in training and 5.06% in validation.

## 5. Balancing the training set

The validation set was kept unbalanced.

In the main flow the training set was balanced with two operations:

1. **Oversampling of positives:** the 35,841 pneumothorax-positive radiographs were concatenated a second time, giving 71,682 positive entries. Oversampling was therefore done by duplicating the same images, not by generating new cases.
2. **Undersampling of negatives:** 286,728 negative images were kept, giving approximately four negatives for each positive.

The resulting checkpoint, `all_train_df_balanced_ckpt3.csv`, contains 358,410 entries (322,569 unique images):

| Pneumothorax class | Entries | Proportion |
|---|---:|---:|
| Negative | 286,728 | 80% |
| Positive | 71,682 | 20% |
| **Total** | **358,410** | **100%** |

The file contains 35,841 duplicated paths, corresponding to the oversampling of all positives.

The notebook also created an alternative, `all_train_df_balanced_ckpt3_2.csv`, with 409,442 entries and a positive proportion of 17.5%. This was a later experimental branch and was not used for the released model.

## 6. Image preparation

Images were loaded with `ImageDataGenerator.flow_from_dataframe` using:

- image column: `filepath`;
- labels: `pneumothorax`, `support_devices` and `no_finding`;
- input size: 456 × 456 pixels;
- RGB color mode (three channels);
- multi-label format (`class_mode='raw'`);
- shuffling for the training set;
- fixed order, no shuffling, for validation;
- seed 159 for the generators.

No explicit division by 255 was applied in the generator: the loaded EfficientNet contains its own initial rescaling and normalization layers.

### Data augmentation

Augmentation was applied only during training:

- random rotation of up to 10°;
- horizontal flip;
- horizontal shift between −3% and +3%;
- vertical shift between −3% and +3%;
- brightness change between 0.8 and 1.2;
- shear of up to 3°;
- nearest-neighbor pixel filling.

No vertical flips were used. Validation used no augmentation.

The generators were wrapped in `tf.data.Dataset` objects with `prefetch(AUTOTUNE)`.

## 7. Model architecture

The model `effb5_cxr_domain_pretraining.keras` was loaded: an EfficientNet-B5 previously adapted to the chest radiograph domain.

Its last classification layer was removed and a dense layer with three units and sigmoid activation was added:

```text
Pretrained EfficientNet-B5
          ↓
representation before the original head
          ↓
Dense(3, activation="sigmoid")
          ↓
pneumothorax | support_devices | no_finding
```

The three outputs are interpreted independently, as corresponds to a multi-label problem.

## 8. Loss function and metrics

A weighted binary cross-entropy was used. The function computed the binary cross-entropy of each of the three outputs, multiplied each one by a weight and averaged the three values.

The weights did not distinguish between positive and negative examples within each label; they weighted the overall contribution of each task. During progressive fine-tuning the following were tried:

- `[4, 1, 0.8]`;
- `[5, 1, 1]`;
- `[5, 0.8, 0.8]`;
- `[6, 0.8, 0.8]`;
- `[8, 0.8, 0.8]`.

The first value corresponds to pneumothorax, which therefore received an increasing weight.

The metrics logged during training were:

- binary accuracy;
- precision;
- sensitivity (recall);
- multi-label ROC AUC;
- weighted F1 at a 0.5 threshold;
- pneumothorax-specific AUC.

The optimizer was Adam.

## 9. Progressive fine-tuning

Training followed a progressive unfreezing strategy. Instead of fine-tuning the whole EfficientNet from the start, increasingly deeper blocks were released.

### Phase 1: block 7

- Most of the EfficientNet was frozen.
- Layers from the beginning of block 7 and the new head were made trainable.
- Learning rate: 2 × 10⁻⁴.
- Task weights: `[4, 1, 0.8]`.
- The first complete epoch recorded a global validation AUC of 0.8791 and a pneumothorax validation AUC of 0.8750.

### Phase 2: block 6

- Layers from block 6 were unfrozen.
- Learning rate: 5 × 10⁻⁵.
- Weights: `[5, 1, 1]`.
- Two epochs were run.
- At the end the global validation AUC was 0.9054 and the pneumothorax validation AUC 0.9154.

### Phase 3: block 5

- Layers from block 5 were unfrozen.
- Initial learning rate: 1 × 10⁻⁵.
- Weights `[5, 0.8, 0.8]` were used first.
- The documented epoch recorded a global validation AUC of 0.9064 and a pneumothorax AUC of 0.9180.

### Phase 4: higher pneumothorax weight

- Block 5 remained unfrozen.
- Learning rate: 1 × 10⁻⁵.
- Weights: `[6, 0.8, 0.8]`.
- Additional passes were run.
- The best documented pass recorded a global validation AUC of 0.9091 and a pneumothorax validation AUC of 0.9220.
- The weights were saved as `best_model_5_2.keras`. **This is the released multisource model.**

### Phase 5: additional trial and overfitting

- Block 5 remained unfrozen.
- The learning rate was reduced to 2.5 × 10⁻⁶.
- The weights were changed to `[8, 0.8, 0.8]`.
- The pneumothorax validation AUC dropped to 0.9199.
- The notebook itself noted that `best_model_5_3` was worse because it had started to overfit.

The checkpoint selected as the best overall model was therefore `best_model_5_2.keras`, prior to this last phase.

## 10. Callbacks and logging

The following were used:

- `WandbMetricsLogger`;
- `ModelCheckpoint`;
- `ReduceLROnPlateau`;
- `EarlyStopping` with restoration of the best weights.

In different phases `val_auc` or `auc_pneumothorax` were monitored. Models were saved with names associated with the first unfrozen block, for example `best_model_7.keras`, `best_model_6.keras`, `best_model_5.keras`, `best_model_5_2.keras` and `best_model_5_3.keras`.

## 11. Evaluation on the main validation set

The main preserved evaluation used the 27,535 validation images and three labels.

### Fixed threshold of 0.5

For pneumothorax:

- ROC AUC: 0.9167;
- average precision: 0.5277;
- precision: 0.3831;
- sensitivity: 0.6683;
- F1: 0.4871;
- true positives: 931;
- false positives: 1,499;
- false negatives: 462;
- true negatives: 24,643.

### Threshold maximizing F1 on validation

The threshold computed for pneumothorax was approximately 0.636. With it:

- precision: 0.4761;
- sensitivity: 0.5779;
- F1: 0.5220;
- true positives: 805;
- false positives: 886;
- false negatives: 588;
- true negatives: 25,256.

### Manual sensitivity-oriented threshold

A manual pneumothorax threshold of 0.20 was also tried. With it:

- precision: 0.2136;
- sensitivity: 0.8363;
- F1: 0.3402;
- true positives: 1,165;
- false positives: 4,290;
- false negatives: 228;
- true negatives: 21,852.

ROC and precision–recall curves, confusion matrices, threshold sweeps and a visual review of false positives and false negatives were also produced. The complete tables and figures of the final model on validation, on the public external test sets and on the SMS clinical dataset are in [`results/`](../results/).

## 12. Later experimental branches (discarded)

After the main training, experiments were run using data from the validation set:

1. direct training on the 27,535 validation images;
2. training on a balanced validation subset of 5,893 images (4,500 negatives and 1,393 positives);
3. training on 63,376 images, combining the whole validation set with the original training positives.

The notebook includes a note stating that this retraining with poorly prepared, highly unbalanced validation data harmed the model; the weights `best_model_5_2.keras` were then reloaded as the best general weights available.

These branches must be considered discarded experiments. Had they been used to produce the final model, the original validation set would no longer be valid for estimating performance, and evaluation would have to rely exclusively on independent external data.

## 13. Older results in the same notebook

The notebook also contains an evaluation on 2,110 images, with 422 pneumothorax-positive cases and an AUC of 0.9486. The Weights & Biases output links it to a run named `NIH - PTX multilabel`.

This block comes from another session or an earlier state of the notebook and does not correspond to the validation on the 27,535 images of the six datasets. It must not be combined with the metrics of the main flow or presented as its final evaluation.

## Overview

```text
Six public datasets
904,454 images
        │
        ├── Metadata harmonization
        ├── AP/PA selection
        └── Exclusion of lateral and unclassified projections
        ↓
687,864 frontal radiographs
        │
        ├── Harmonization of the three labels
        └── Patient-level split stratified within each dataset
        ↓
Original training: 660,329          Validation: 27,535
        │                                  │
        ├── Duplicate PTX positives        └── No balancing, no augmentation
        └── Undersample negatives 4:1
        ↓
Balanced training: 358,410
        │
        ├── Resize to 456 × 456 RGB
        ├── Data augmentation
        └── EfficientNet-B5 pretrained on chest radiographs
        ↓
3-output sigmoid head
        │
        └── Progressive fine-tuning, blocks 7 → 6 → 5
        ↓
Selected checkpoint: best_model_5_2.keras  (released multisource model)
        │
        ├── Multi-label evaluation and threshold selection (validation)
        ├── Public external test: BRAX, CANDID-PTX, CRADI, Indiana, PTX-498
        └── External validation on the private SMS clinical dataset
```

## External evaluation

The released models were evaluated, without further fine-tuning, on:

- **Public external test sets** (BRAX, CANDID-PTX, CRADI, Indiana and PTX-498; 53,426 images after filtering projections and problematic files). Results: [`results/public_external_test/`](../results/public_external_test/), notebook `5.4-test`.
- **SMS clinical dataset** (Servicio Murciano de Salud, the Murcia Health Service, Spain; 202 images, 112 with pneumothorax), a private dataset that is not distributed with this repository. Results: [`results/sms_external_validation/`](../results/sms_external_validation/), notebook `5.4.2-exttest` (it requires the private data to run).

## Reproducibility notes

Points to keep in mind when reproducing or interpreting the notebooks:

1. **Filter of problematic files in the evaluation notebooks.** The evaluation notebooks assign two consecutive filters:

   ```python
   df = ds.df.loc[~ds.df.filepath.isin(bad_study_filepaths)]
   df = ds.df.loc[~ds.df.filepath.isin(bad_lat_filepaths)]
   ```

   The second statement starts again from `ds.df`, so the first filter has no effect as written; only `bad_lat_filepaths` is applied there.
2. **Uncertain and missing labels.** Uncertain and missing pneumothorax values were treated as negatives, as were missing values of the auxiliary labels. This can introduce systematic label noise.
3. **Unreadable image files.** During training, images that could not be loaded were repeatedly reported and replaced by a dummy image so that training was not interrupted, and the generators used `validate_filenames=False`. The number of such files was not recorded.
4. **Seeds.** Several seeds appear: `config.seed = 168` (stored in the run configuration and logged to Weights & Biases, but not passed to any random number generator in the notebooks), 159 (image generators, applied), 42 (patient split, default of its function, applied) and no `random_state` for the negative undersampling (`undersample_label` calls `df.sample(n)` and therefore uses the unseeded global NumPy generator), so the exact negative sample cannot be regenerated with the current code. The exact list of training images is therefore given by the checkpoint CSV, summarized in `splits/development_splits.csv.gz`.
5. **Batch size.** Batch sizes of 6, 4 and 8 were assigned successively; different phases used different sizes.
6. **Checkpoints.** `ModelCheckpoint` was configured with `save_best_only=False`, so every epoch was saved and files with the same name could be overwritten; two different cells save to `best_model_5_2.keras`. Relative paths based on `wandb/latest-run` were used. The released files are identified by their SHA-256 in [`MODELS.md`](../MODELS.md).
7. **Monitored metrics.** In the first phases `val_auc` was monitored, but in several later phases `ReduceLROnPlateau` and `EarlyStopping` monitored `auc_pneumothorax` without the `val_` prefix, i.e. the training metric rather than the validation metric.
8. **Threshold optimization.** The F1-maximizing thresholds were computed and evaluated on the same validation set, so the metrics obtained with those thresholds are optimistic as a final estimate.
