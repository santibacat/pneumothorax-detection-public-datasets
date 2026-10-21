# Pneumothorax detection on public chest X-ray datasets

Code, trained models and results for the paper:

> Ibañez Caturla, S.; Berná Mestre, J.D.; Martinez Mozos, O. **Towards Automatic Detection of Pneumothorax in Emergency Care with Deep Learning Using Multi-Source Chest X-ray Data.** *Future Internet* 2025, 17(7), 292. [https://doi.org/10.3390/fi17070292](https://doi.org/10.3390/fi17070292) ([MDPI](https://www.mdpi.com/1999-5903/17/7/292))

We train multi-label EfficientNet-B5 models (`pneumothorax`, `support_devices`, `no_finding`) on frontal chest radiographs from six public datasets (NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest, VinDr-CXR and SIIM-ACR) and evaluate them on held-out data and on external datasets never seen during training. We release one **multisource** model trained on all six datasets and six **singlesource** models trained on each dataset alone, which allows measuring how well models transfer across data sources.

## Evaluation

Pneumothorax results for the multisource model (AUC and average precision with 95% confidence intervals; full results for every model, class and threshold are in [`results/`](results/)):

| Evaluation set | Images (pneumothorax) | AUC | Average precision |
|---|---:|---|---|
| Validation (held-out patients from the six training datasets) | 27,535 (1,393) | 0.922 (0.914–0.929) | 0.552 (0.523–0.578) |
| Public external test: BRAX, CANDID-PTX, CRADI, Indiana, PTX-498 | 53,426 (4,081) | 0.961 (0.958–0.965) | 0.824 (0.814–0.834) |
| **External validation on the SMS (Servicio Murciano de Salud) clinical dataset** | 202 (112) | 0.886 (0.836–0.929) | 0.902 (0.843–0.948) |

The SMS dataset comes from the **Servicio Murciano de Salud (SMS, the Murcia Health Service, Spain)**. It is a **private clinical dataset that is not part of this repository** and cannot be redistributed. Its aggregated results (metrics and figures, no images or patient data) are included in [`results/sms_external_validation/`](results/sms_external_validation/), and the notebook that produced them, [`notebooks/evaluate/5.4.2-exttest PTX BY DATASET multilabel EffB5.ipynb`](notebooks/evaluate), is provided for transparency but requires that private data to run. All other evaluations can be reproduced with the public datasets (see below).

## Repository contents

| Path | Description |
|---|---|
| `cxr/` | Python package: dataset loaders, training and evaluation code |
| `notebooks/00_quickstart.ipynb` | Usage guide and environment check: load datasets, link them to `splits/`, run a model |
| `notebooks/train/` | `5.2` multisource model; `5.3.x` one singlesource model per dataset |
| `notebooks/evaluate/` | `5.4.1` validation, `5.4` public external test, `5.4.2` SMS external validation |
| `splits/` | Train/validation assignment and external test image lists (identifiers only, see [`splits/README.md`](splits/README.md)) |
| `results/` | Metrics (CSV) and figures for `validation`, `public_external_test` and `sms_external_validation` |
| `docs/methodology.md` | Detailed description of the training procedure |
| `docs/datasets.md` | Expected folder layout of the datasets and what to check when loading fails |
| `MODELS.md` | Model download links, Weights & Biases run ids and checksums |

## Getting started

1. Install the environment with [uv](https://docs.astral.sh/uv/): `uv sync` (Python 3.10, TensorFlow 2.15.1; on Linux the CUDA libraries are installed too, so an NVIDIA GPU is used automatically). This also installs the `cxr` package and JupyterLab.
2. Download the original datasets and arrange them as described in [`docs/datasets.md`](docs/datasets.md). Several require credentialed access (MIMIC-CXR, PadChest, CheXpert, VinDr-CXR, ...).
3. Download the models listed in [`MODELS.md`](MODELS.md) into `models/`.
4. Set `CXR_DATASETS_PATH` to the folder that contains your datasets (or edit `cxr/config.py`) and start JupyterLab:

   ```bash
   CXR_DATASETS_PATH=/data/ds-CXR uv run jupyter lab
   ```

   Start with [`notebooks/00_quickstart.ipynb`](notebooks/00_quickstart.ipynb), which checks that everything works. The other notebooks load the weights from `models/`; the first cell of each notebook sets the repository root as working directory.

## Data availability

The images and labels of the public datasets are **not redistributed**; they must be obtained from their original sources under their own licenses. This repository only provides image identifiers and split assignments (`splits/`). The SMS clinical dataset is private and is not included.

## License

- Code: Apache License 2.0 (see [`LICENSE`](LICENSE)).
- Model weights: CC BY-NC 4.0 (non-commercial use), because they were trained on datasets whose licenses restrict commercial use (see [`MODELS.md`](MODELS.md)).
- The original datasets keep their own licenses.

## Citation

If you use this code or the models, please cite the paper above. See also [`CITATION.cff`](CITATION.cff).

```bibtex
@article{ibanez2025pneumothorax,
  author  = {Ibañez Caturla, Santiago and Berná Mestre, Juan de Dios and Martinez Mozos, Oscar},
  title   = {Towards Automatic Detection of Pneumothorax in Emergency Care with Deep Learning Using Multi-Source Chest X-ray Data},
  journal = {Future Internet},
  year    = {2025},
  volume  = {17},
  number  = {7},
  pages   = {292},
  doi     = {10.3390/fi17070292}
}
```
