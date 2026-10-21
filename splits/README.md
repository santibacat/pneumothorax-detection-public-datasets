# Splits

These files contain **only identifiers and split assignments**: no labels, age, sex or dates. Paths are relative to the root folder of your datasets (`CXR_DATASETS_PATH`, see `cxr/config.py`). Labels must be obtained from the original datasets, e.g. through the loaders in the `cxr` package.

## `development_splits.csv.gz`

687,864 frontal (AP/PA) radiographs from NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest, VinDr-CXR and SIIM-ACR, after projection filtering.

| Column | Description |
|---|---|
| `dataset` | Source dataset |
| `patient_uuid`, `study_uuid` | Patient and study identifiers (`<dataset>_<id>`) |
| `filepath` | Image path relative to the datasets root |
| `split` | `train` (660,329 images) or `valid` (27,535 images) |
| `in_multisource_training_set` | `True` for the images that were used to train the multisource model |
| `in_singlesource_training_set` | `True` for the images used to train the singlesource model of the image's own dataset |

The split is stratified by patient on pneumothorax presence, independently within each dataset (no patient appears in both `train` and `valid`). The multisource model was trained on a rebalanced version of the training split: every pneumothorax-positive image was used twice and the negatives were undersampled to a 4:1 ratio, giving 358,410 training entries from 322,569 unique images. `in_multisource_training_set` marks those unique images (see [`docs/methodology.md`](../docs/methodology.md)).

## Which images each model used

| Model | Training images | Validation images |
|---|---|---|
| Multisource | `split == "train"` and `in_multisource_training_set` (358,410 entries, 322,569 unique images, positives used twice) | `split == "valid"` (27,535) |
| Singlesource (NIH, CheXpert, MIMIC-CXR, PadChest, SIIM-ACR, VinDr-CXR) | `in_singlesource_training_set` for the model's own dataset. Entries (positives used twice) / unique images: NIH 65,647 / 60,611; CheXpert 110,759 / 93,797; MIMIC-CXR 123,600 / 112,806; PadChest 49,493 / 49,100; SIIM-ACR 9,275 / 6,703; VinDr-CXR 8,090 / 8,006 | `split == "valid"` images of the model's own dataset |

All models share the same patient-level `train`/`valid` assignment; the singlesource models use the part of it that belongs to their dataset. The training lists come from the checkpoint files of the runs: the multisource model and the CheXpert, MIMIC-CXR, PadChest, SIIM-ACR and VinDr-CXR singlesource models used the rebalanced file `all_train_df_balanced_ckpt3.csv` (filtered by dataset for the latter), while the NIH singlesource model used `all_train_df_balanced_ckpt3_2.csv` filtered to NIH. The training sizes match the ones reported in the logs of the runs. The negatives were selected by random undersampling with the global (unseeded) NumPy generator, so the selection cannot be regenerated with the code, but it is recorded in these flags.

## `external_test.csv.gz`

53,426 frontal radiographs from the public external test datasets BRAX, CANDID-PTX, CRADI, Indiana and PTX-498, which were not used for training. Columns: `dataset`, `patient_uuid`, `study_uuid`, `filepath`.

The private SMS clinical dataset used for external validation is not part of this repository, so no image list is provided for it.
