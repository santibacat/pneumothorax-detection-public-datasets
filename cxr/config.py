import os

# LOCATION OF DATASETS
# Don't use last slash /
# Root folder with all the datasets. Set the CXR_DATASETS_PATH env variable
# (or edit this line) to point to your local copy.
DATASETS_PATH = os.environ.get("CXR_DATASETS_PATH", "/path/to/ds-CXR")
NIH_PATH = f"{DATASETS_PATH}/nih"
CHEXPERT_PATH = f"{DATASETS_PATH}/CheXpert-v1.0"
MIMICCXR_PATH = f"{DATASETS_PATH}/MIMIC-CXR-JPG/2.0.0/"
PADCHEST_PATH = f"{DATASETS_PATH}/padchest/"
VINDRCXR_PATH = f"{DATASETS_PATH}/VinDR-CXR/"
BRAX_PATH = f"{DATASETS_PATH}/BRAX/1.1.0/"
PEDIATRICPNEUMONIA_PATH = f"{DATASETS_PATH}/chest-xrays-pneumonia"
JSRT_PATH = f"{DATASETS_PATH}/jsrt"
SIIM_PNEUMOTHORAX_PATH = f"{DATASETS_PATH}/siim-acr-pneumothorax-segmentation/abishek-siim-png-images/input"
PTX498_PATH = f"{DATASETS_PATH}/PTX-498/"
CANDIDPTX_PATH = f"{DATASETS_PATH}/CANDID-PTX/14909820/"
CRADI_PATH = f"{DATASETS_PATH}/CRADI/"
BELARUS_PATH = f"{DATASETS_PATH}/belarus"
OBJECT_CXR_PATH = (
    f"{DATASETS_PATH}/kaggle-Foreign object CXR/object-CXR"
)
MONTGOMERY_PATH = f"{DATASETS_PATH}/tbc-montgomery"
SHENZHEN_PATH = f"{DATASETS_PATH}/tbc-shenzhen"
TBX11K_PATH = f"{DATASETS_PATH}/TBX11K/TBX11K"
DADB_PATH = f"{DATASETS_PATH}/tbc-DA-DB"
RANZCR_PATH = f"{DATASETS_PATH}/RANZCR-catheter/ranzcr-clip-catheter-line-classification"
INDIANA_PATH = f"{DATASETS_PATH}/indiana"

DATASETS = {
    "NIHDataset": NIH_PATH,
    "CheXpertDataset": CHEXPERT_PATH,
    "MIMICCXRDataset": MIMICCXR_PATH,
    "PadChestDataset": PADCHEST_PATH,
    "VinDrCXRDataset": VINDRCXR_PATH,
    "SIIMACRPneumothoraxDataset": SIIM_PNEUMOTHORAX_PATH,
    "BRAXDataset": BRAX_PATH,
    "CANDIDPTXDataset": CANDIDPTX_PATH,
    "IndianaDataset": INDIANA_PATH,
    "PTX498Dataset": PTX498_PATH,
    "CRADIDataset": CRADI_PATH,
    "JSRTDataset": JSRT_PATH,
    "PediatricPneumoniaDataset": PEDIATRICPNEUMONIA_PATH,
    "BelarusDataset": BELARUS_PATH,
    "ObjectCXRDataset": OBJECT_CXR_PATH,
    "MontgomeryDataset": MONTGOMERY_PATH,
    "ShenzhenDataset": SHENZHEN_PATH,
    "TBX11KDataset": TBX11K_PATH,
    "DADBDataset": DADB_PATH,
    "RANZCRDataset": RANZCR_PATH,
}

PNEUMOTHORAX_DATASETS = {
    "NIHDataset": NIH_PATH,
    "CheXpertDataset": CHEXPERT_PATH,
    "MIMICCXRDataset": MIMICCXR_PATH,
    "PadChestDataset": PADCHEST_PATH,
    "VinDrCXRDataset": VINDRCXR_PATH,
    "SIIMACRPneumothoraxDataset": SIIM_PNEUMOTHORAX_PATH,
    "BRAXDataset": BRAX_PATH,
    "CANDIDPTXDataset": CANDIDPTX_PATH,
    "IndianaDataset": INDIANA_PATH,
    "PTX498Dataset": PTX498_PATH,
    "CRADIDataset": CRADI_PATH,
}

PNEUMOTHORAX_TRAINING_DATASETS = {
    "NIHDataset": NIH_PATH,
    "CheXpertDataset": CHEXPERT_PATH,
    "MIMICCXRDataset": MIMICCXR_PATH,
    "PadChestDataset": PADCHEST_PATH,
    "VinDrCXRDataset": VINDRCXR_PATH,
    "SIIMACRPneumothoraxDataset": SIIM_PNEUMOTHORAX_PATH,
}

PNEUMOTHORAX_TESTING_DATASETS = {
    "BRAXDataset": BRAX_PATH,
    "CANDIDPTXDataset": CANDIDPTX_PATH,
    "IndianaDataset": INDIANA_PATH,
    "PTX498Dataset": PTX498_PATH,
    "CRADIDataset": CRADI_PATH,
}
