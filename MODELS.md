# Trained models

All models are EfficientNet-B5 networks in Keras format (TensorFlow 2.15.1). `effb5_cxr_domain_pretraining` is the chest-radiograph pretrained backbone from which every other model starts. The `effb5_ptx_*` models are multi-label (`pneumothorax`, `support_devices`, `no_finding`): one **multisource** model trained on the six development datasets and six **singlesource** models trained on each dataset alone.

**Download:** all models are in this public, read-only [Google Drive folder](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing). Put the files into `models/` keeping the names below, and check them with `sha256sum models/*.keras`.

| File | id | Description | Size | SHA-256 | Download |
|---|---|---|---:|---|---|
| `effb5_cxr_domain_pretraining.keras` | `tfy36ftx` | Domain-adapted pretraining on chest radiographs (starting point of all the other models) | 300 MB | `ea513e23576c39f01bf862a366757a59a4376520c5baf878723f02311dee591c` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |
| `effb5_ptx_multisource.keras` | `771kvn05` | Multisource: trained on the six development datasets | 326 MB | `f9067916ee76f8259fcb3b62fde92338c695a9ca77d119a325d575af65b68197` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |
| `effb5_ptx_singlesource_chexpert.keras` | `z31hcpn7` | Singlesource: trained only on CheXpert | 326 MB | `6a1b54e44059ae9c1f256f6426ce7cdcbd91a64244112d697998a7daeb613792` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |
| `effb5_ptx_singlesource_mimic.keras` | `v3azo7z0` | Singlesource: trained only on MIMIC-CXR | 326 MB | `9cba6d57a7a14e04e095378323d836cf8c6e3bd2490d6c57ec354863517d3e81` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |
| `effb5_ptx_singlesource_nih.keras` | `obec6sw2` | Singlesource: trained only on NIH | 326 MB | `f25b3bbf93576a12072c77bce71e9ed785c4087822730cee381cfdd3c15cdd06` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |
| `effb5_ptx_singlesource_padchest.keras` | `5nn8el06` | Singlesource: trained only on PadChest | 326 MB | `20a5289fa923f302c4910f1603ab81ce7423c99c875e00a37e29a76f236bef22` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |
| `effb5_ptx_singlesource_siim.keras` | `b5h7q4r1` | Singlesource: trained only on SIIM-ACR | 326 MB | `7ae7177ea50844e015385233797878ae2eb8c963b73b4fce6ba3f0c6063f73b3` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |
| `effb5_ptx_singlesource_vindr.keras` | `nncp4yw8` | Singlesource: trained only on VinDr-CXR | 326 MB | `ba7ba28e684f61e4e63f4af95072aa89b490d4857d3d512890095f87edbc9abf` | [Drive](https://drive.google.com/drive/folders/11dAA5eXilMS_sIVTcBRems-YSAC4xHmK?usp=sharing) |

- `id` is the run identifier in Weights & Biases (project `santibacat/cxr`). The multisource model corresponds to the checkpoint `best_model_5_2` of its run.
- Input: RGB image of 456 × 456 pixels with values in 0–255 (rescaling is part of the model). Output: three sigmoid probabilities in the order `pneumothorax`, `support_devices`, `no_finding`.
- Load with `tf.keras.models.load_model("models/effb5_ptx_multisource.keras", compile=False)`. See [`notebooks/00_quickstart.ipynb`](notebooks/00_quickstart.ipynb).

## License

The model weights are released under **CC BY-NC 4.0** (non-commercial use only), because they were trained on datasets whose licenses restrict commercial use (e.g. CheXpert, PadChest). The code is released under Apache-2.0 (see `LICENSE`). Users are responsible for complying with the licenses of the original datasets.
