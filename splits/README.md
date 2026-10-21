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

The split is stratified by patient on pneumothorax presence, independently within each dataset (no patient appears in both `train` and `valid`). The multisource model was trained on a rebalanced version of the training split: every pneumothorax-positive image was used twice and the negatives were undersampled to a 4:1 ratio, giving 358,410 training entries from 322,569 unique images. `in_multisource_training_set` marks those unique images (see [`docs/methodology.md`](../docs/methodology.md)).

## Which images each model used

| Model | Training images | Validation images |
|---|---|---|
| Multisource | Exactly identified: `split == "train"` and `in_multisource_training_set` (358,410 entries, 322,569 unique images, positives used twice) | `split == "valid"` (27,535) |
| Singlesource (NIH, CheXpert, MIMIC-CXR, PadChest, SIIM-ACR, VinDr-CXR) | A rebalanced subset of the `train` images of the model's own dataset. Only the sizes are known (generator logs): NIH 65,647; CheXpert 110,759; MIMIC-CXR 123,600; PadChest 49,493; SIIM-ACR 9,275; VinDr-CXR 8,090 | `split == "valid"` images of the model's own dataset |

For the multisource model the training list comes directly from the checkpoint file used by the run. For the singlesource models the negatives were randomly undersampled with the global (unseeded) NumPy generator, so the exact subset was not saved and cannot be regenerated, so only the dataset-level `train`/`valid` assignment is available; the validation images and the evaluation sets are exactly identified for all models.

## `external_test.csv.gz`

53,426 frontal radiographs from the public external test datasets BRAX, CANDID-PTX, CRADI, Indiana and PTX-498, which were not used for training. Columns: `dataset`, `patient_uuid`, `study_uuid`, `filepath`.

The private SMS clinical dataset used for external validation is not part of this repository, so no image list is provided for it.
