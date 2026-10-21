import os
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed


#####
# from dotenv import load_dotenv
# load_dotenv()
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
###

from cxr.logger import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from cxr.config import (
    NIH_PATH,
    CHEXPERT_PATH,
    MIMICCXR_PATH,
    PADCHEST_PATH,
    VINDRCXR_PATH,
    BRAX_PATH,
    PEDIATRICPNEUMONIA_PATH,
    JSRT_PATH,
    SIIM_PNEUMOTHORAX_PATH,
    PTX498_PATH,
    CANDIDPTX_PATH,
    CRADI_PATH,
    BELARUS_PATH,
    OBJECT_CXR_PATH,
    MONTGOMERY_PATH,
    SHENZHEN_PATH,
    DATASETS,
    TBX11K_PATH,
    DADB_PATH,
    RANZCR_PATH,
    INDIANA_PATH,
)
from PIL import Image

from glob import glob
from pathlib import Path
from itertools import chain
import logging
import pandas as pd
import numpy as np
import ast
import xmltodict
import pydicom
from tqdm import tqdm
# import tensorflow as tf

from cxr.utils import *
from cxr.data_changes.data_changes import find_corrupted_images_parallel
from cxr.mappings import (
    nih_column_mapping,
    chexpert_column_mapping,
    padchest_column_mapping,
    cradi_column_mapping,
    brax_column_mapping,
    vindr_column_mapping,
    label_mapping,
    OPACITY_LABELS,
    mimic_column_mapping,
)


def load_datasets(
    datasets_to_load: Dict[str, str] = DATASETS, combine: bool = True
):
    """If combine, returns a single dataset with all the data. Otherwise, returns a list of datasets"""
    datasets = [
        eval(f"{dataset}()")
        for dataset, path in datasets_to_load.items()
        if path is not None
    ]
    for dataset in datasets:
        logging.info(f"Loading dataset {dataset.get_dataset_name()}...")
        try:
            dataset.load()
        except Exception as e:
            logger.error(f"Error loading {dataset.get_dataset_name()}: {e}")
            # TODO: Eliminar este dataset de la lista
    if combine:
        return sum(datasets)
    else:
        return datasets


class CXRDataset:
    dataset_sources = []
    dataset_labels = []

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.df = pd.DataFrame()  # Initialize an empty DataFrame
        if self.get_dataset_name() is not None:
            self.__class__.dataset_sources.append(self.get_dataset_name())

    def __add__(self, other_dataset):
        if not isinstance(other_dataset, CXRDataset):
            raise ValueError("Both objects must be instances of CXRDataset")

        new_dataset = CXRDataset()

        try:
            # Reset indices of both DataFrames before concatenation
            self.df = self.df.reset_index(drop=True)
            other_dataset.df = other_dataset.df.reset_index(drop=True)

            # Concatenate DataFrames
            new_dataset.df = pd.concat(
                [self.df, other_dataset.df], ignore_index=True
            )

            # Combine dataset labels without duplicates
            new_dataset.dataset_sources = list(
                set(self.dataset_sources + other_dataset.dataset_sources)
            )
            new_dataset.dataset_labels = list(
                set(self.dataset_labels + other_dataset.dataset_labels)
            )
            new_dataset.dataset_path = None
        except Exception as e:
            logger.error(f"Error concatenating datasets: {e}")
            raise (e)
            new_dataset.df = (
                self.df.copy()
            )  # Fallback to copying the current DataFrame

        return new_dataset

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @classmethod
    def get_dataset_name(cls):
        return "combined_cxr"

    def __str__(self):
        return f"CXRDataset with data from: {', '.join(self.dataset_sources)}"

    @classmethod
    def get_dataset_sources(cls):
        return list(set(cls.dataset_sources))

    def get_dataset_labels(self):
        return list(set(self.dataset_labels))

    def load(self, suffix: str = ""):
        try:
            self.load_processed_dataset(suffix)
        except Exception:
            logger.info(f"Loading dataset with {suffix=}...")
            self.load_dataset()
            logger.info("Preprocessing dataset...")
            self.preprocess_dataset()

    def load_processed_dataset(self, suffix: str = ""):
        if self.dataset_path is None:
            raise ValueError(
                "Dataset path is not set. If this CXRDataset is a combination of datasets, you can't use this method."
            )
        try:
            self.df = pd.read_csv(
                os.path.join(
                    self.dataset_path, f"processed_dataset{suffix}.csv"
                )
            )
            logging.info(
                f"Processed dataset {suffix} loaded from {self.dataset_path}"
            )
        except FileNotFoundError as e:
            logger.warning(
                "Processed dataset not found. You need to run load_dataset() and preprocess_dataset() and then save_processed_dataset() to create it."
            )
            raise (e)

        except Exception as e:
            logger.error(f"Error loading processed dataset: {e}")
            raise (e)

    def save_processed_dataset(self, suffix: str = ""):
        if self.dataset_path is None:
            raise ValueError(
                "Dataset path is not set. If this CXRDataset is a combination of datasets, you can't use this method."
            )
        self.df.to_csv(
            os.path.join(self.dataset_path, f"processed_dataset{suffix}.csv"),
            index=False,
        )

    def load_dataset(self):
        pass

    def preprocess_dataset(self):
        pass

    def preprocessing_common(self):
        self.df.loc[:, "patient_uuid"] = (
            self.get_dataset_name() + "_" + self.df["patient_id"].astype("str")
        )
        self.df.loc[:, "study_uuid"] = (
            self.get_dataset_name()
            + "_"
            + self.df["patient_id"].astype("str")
            + "_"
            + self.df["study_id"].astype("str")
        )

    def describe_dataset(self):
        print(
            f"Dataset LABEL: {self.get_dataset_name() if self.get_dataset_name() is not None else 'Custom CXR Dataset'}"
        )
        if self.dataset_path is not None:
            print(f"Dataset PATH: {self.dataset_path})")
        print(
            f"IMAGES from each df: \n{self.df.dataset.value_counts(dropna=False)}\n"
        )
        print(f"Unique patients: {self.df.patient_uuid.unique().shape[0]}\n")
        try:
            print(
                f"Patients' age: {round(self.df.patient_age.mean(), 2)} +- {round(self.df.patient_age.std(), 2)}\n"
            )
            print(
                f"Patients' sex:\n{self.df.patient_sex.value_counts(dropna=False)}\n"
            )
        except Exception as e:
            print(f"Error calculating age and sex {e}")
        # print(f"Unique labels: {self.df.labels.unique().shape[0]}")
        # print(f"Unique followups:\n{self.df.followup.value_counts(dropna=False)}")
        try:
            print(
                f"Original data SPLIT: \n{self.df.test_split.value_counts(dropna=False, normalize=True)}\n"
            )
        except:
            logger.debug("No data split")

    def verify_dataset(self):
        print(f"Largest ages: {self.df.patient_age.nlargest(20).values}")
        print(f"Smallest ages: {self.df.patient_age.nsmallest(20).values}")

    def combine_datasets(self, *dfs):
        self.combined_df = pd.concat([*dfs], ignore_index=True)

    def remove_unnamed_columns(self, df):
        df.drop(
            columns=[x for x in df.columns if x.startswith("Unnamed")],
            inplace=True,
        )

    def sample(self, n):
        return self.df.sample(n)

    def show_sample(self, n):
        """Show sample of images from column filepath"""
        sample = self.df.sample(n)
        for index, row in sample.iterrows():
            print(f"{row}")
            img = Image.open(row["filepath"])
            display(img)

    def show_sample_datagen(
        self,
        n,
        mode="binary",
        color_mode="grayscale",
        target_size=(224, 224),
        class_mode=None,
    ):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from cxr.utils import show_generator

        datagen = ImageDataGenerator(rescale=1.0 / 255)
        generator = datagen.flow_from_dataframe(
            self.df.sample(n),
            x_col="filepath",
            y_col=None,
            target_size=target_size,
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=n,
        )
        show_generator(generator, num_batches=1, mode=mode)

    def label_subset(self, columns, drop=False):
        labels = list(set(self.dataset_labels))
        if "abnormal" in labels:
            labels.remove("abnormal")
        for label in columns:
            if label not in labels:
                raise ValueError(f"Label {label} not in labels")

        for elem in columns:
            if elem in labels:
                labels.remove(elem)

        if drop:
            drop_columns = columns
        else:
            drop_columns = labels
        df = self.df.drop(columns=drop_columns, errors="ignore")

        return df
        # Still need to verify that we have elemens in all columns, if not, remove that row

    def normal_or_abnormal(self, columns):
        pass

    def calculate_no_finding(self):
        """
        Calculate 'no_finding' column based on other label values.
        Sets no_finding to 1 if all other labels are 0, otherwise 0.
        """
        # Get all label columns except 'no_finding'
        label_columns = [
            col for col in self.get_dataset_labels() if col != "no_finding"
        ]

        # Calculate no_finding: 1 if sum of all other labels is 0, else 0
        self.df["no_finding"] = (
            self.df[label_columns].sum(axis=1) == 0
        ).astype(int)

        return self.df.no_finding.value_counts(dropna=False)

    def main(self):
        pass

    def check_corrupted_images(self):
        """
        Check for corrupted images in the dataset using the filepath column.
        Adds a 'bad_image' column to the DataFrame with 1 for corrupted images and 0 for valid ones.

        Returns:
            list: List of corrupted image filepaths
        """
        if "filepath" not in self.df.columns:
            logger.error("No 'filepath' column found in the dataset")
            return None

        # Get list of files from the filepath column
        list_of_files = self.df["filepath"].tolist()

        # Check for corrupted images
        corrupted_images = find_corrupted_images_parallel(list_of_files)

        # Add 'bad_image' column to the DataFrame
        self.df["bad_image"] = 0
        self.df.loc[
            self.df["filepath"].isin(corrupted_images), "bad_image"
        ] = 1

        logger.info(
            f"Found {len(corrupted_images)} corrupted images out of {len(list_of_files)} total images"
        )

        return corrupted_images

    def get_column_nulls_by_dataset_for_single_column(self, column):
        """
        Get the number of nulls for a column by dataset
        """
        return self.df.groupby("dataset")[column].apply(
            lambda x: x.isna().sum() / len(x)
        )

    def get_column_nulls_by_dataset(self):
        for column in self.df.columns:
            try:
                print(
                    f"{self.get_column_nulls_by_dataset_for_single_column(column)}\n\n"
                )
            except Exception as e:
                print(f"Error getting nulls for column {column}: {e}\n\n")

    def get_column_values_by_dataset_for_single_column(self, column):
        return self.df.groupby("dataset")[column].apply(
            lambda x: x.value_counts(dropna=False)
        )

    def get_column_values_by_dataset(self):
        for column in self.df.columns:
            try:
                print(
                    f"{self.get_column_values_by_dataset_for_single_column(column)}\n\n"
                )
            except Exception as e:
                print(f"Error getting values for column {column}: {e}\n\n")


class NIHDataset(CXRDataset):
    def __init__(self, dataset_path=NIH_PATH):
        super().__init__(dataset_path)

    def get_dataset_name(cls):
        return "nih"

    def load_dataset(self):
        logger.debug("Importing NIH")
        try:
            self.orig_df = pd.read_csv(
                os.path.join(self.dataset_path, "Data_Entry_2017_v2020.csv")
            )
            logger.debug("Importing model v2020")
        except Exception as e:
            logger.warning(
                f"Error importing model v2020. Trying with v2017: {e}."
            )
            self.orig_df = pd.read_csv(
                os.path.join(self.dataset_path, "Data_Entry_2017.csv")
            )
            logger.debug(
                "Importing model v1. Please consider upgrading to new version"
            )
        self.orig_df_bbox = pd.read_csv(
            os.path.join(self.dataset_path, "BBox_List_2017.csv")
        )

    def integrate_expert_labels(self, type: str = "all"):
        """Must specify type: 'all' for all expert labels or single experts such as 'four', 'all_findings', 'pneumothorax', 'neatx'"""

        # We set split to None if unused; 1 = test set, 0 = train-val expert set
        if "expert_split" not in self.df.columns:
            self.df["expert_split"] = None

        if type == True:
            type = "all"

        if type == "four" or type == "all":
            expert_folder = "four_findings_expert_labels"
            expert_csvs = ["validation_labels.csv", "test_labels.csv"]

            for i, expert_csv in enumerate(expert_csvs):
                expert = pd.read_csv(
                    os.path.join(self.dataset_path, expert_folder, expert_csv)
                )
                expert.rename(columns=nih_column_mapping, inplace=True)

                # Get expert-labelled filenames
                expert_files = expert["filename"].unique()

                # Create a mask for expert-labelled files
                expert_mask = self.df.filename.isin(expert_files)

                # For each condition, update the main dataframe where expert says YES
                for column in [
                    "pneumothorax",
                    "fracture",
                    "nodule_mass",
                    "opacity",
                ]:
                    yes_mask = expert[column] == "YES"
                    self.df.loc[
                        expert_mask
                        & self.df.filename.isin(
                            expert.loc[yes_mask, "filename"]
                        ),
                        column,
                    ] = 1

                # Set no_finding to 0 for any file that has at least one positive finding
                positive_findings = (
                    self.df[expert_mask][
                        ["pneumothorax", "fracture", "nodule_mass", "opacity"]
                    ].sum(axis=1)
                    > 0
                )
                self.df.loc[expert_mask & positive_findings, "no_finding"] = 0

                # Mark these files as being in this expert split
                self.df.loc[expert_mask, "expert_split"] = i

        if type == "all_findings" or type == "all":
            expert_folder = "all_findings_expert_labels"
            expert_csv = "test_labels.csv"
            expert = pd.read_csv(
                os.path.join(self.dataset_path, expert_folder, expert_csv)
            )
            expert.rename(columns=nih_column_mapping, inplace=True)

            # We don't create abnormal column, as it's just when there are any finding apart from no_finding
            # If you need it, just use df['no_finding'] == 0
            self.df["other"] = 0

            # Get expert-labelled filenames
            expert_files = expert["filename"].unique()

            # Create a mask for expert-labelled files
            expert_mask = self.df.filename.isin(expert_files)

            # For each condition, update the main dataframe where expert says YES
            for column in expert.columns[2:]:
                yes_mask = expert[column] == "YES"
                self.df.loc[
                    expert_mask
                    & self.df.filename.isin(expert.loc[yes_mask, "filename"]),
                    column,
                ] = 1

            # If abnormal is true, set no_finding to 0
            self.df.loc[self.df.abnormal == 1, "no_finding"] = 0

            # Mark these files as being in this expert split
            self.df.loc[expert_mask, "expert_split"] = 0
            self.df.drop(columns="abnormal", inplace=True)

        if type == "pneumothorax" or type == "all":
            expert = pd.read_csv(
                os.path.join(
                    self.dataset_path,
                    "NLP_RAD_relabelled_NIH_CXR",
                    "data",
                    "NIH_radiologist_labelled_positive.csv",
                )
            )
            self.df.loc[
                self.df.filename.isin(expert["filename"]), "pneumothorax"
            ] = 1
            self.df.loc[
                self.df.filename.isin(expert["filename"]), "no_finding"
            ] = 0
            self.df.loc[
                ~self.df.filename.isin(expert["filename"]), "pneumothorax"
            ] = 0

        if type == "neatx" or type == "all":
            expert = pd.read_csv(
                os.path.join(
                    self.dataset_path,
                    "neatx",
                    "NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv",
                )
            )
            drain_mapping = dict(zip(expert["Image Index"], expert["Drain"]))
            self.df.loc[
                self.df.filename.isin(drain_mapping), "support_devices"
            ] = self.df.filename.map(drain_mapping)

    def preprocess_dataset(self, use_experts_labels=True):
        self.df = self.orig_df.copy()
        self.df_bbox = self.orig_df_bbox.copy()
        self.df.loc[:, "dataset"] = self.get_dataset_name()
        self.dataset_labels = [
            "mass",
            "pneumonia",
            "emphysema",
            "hernia",
            "cardiomegaly",
            "fibrosis",
            "edema",
            "pleural_thickening",
            "infiltration",
            "no_finding",
            "pneumothorax",
            "nodule",
            "consolidation",
            "atelectasis",
            "pleural_effusion",
            "fracture",
            "nodule_mass",
            "opacity",
        ]

        # Create label mapping
        self.df["labels"] = self.df["Finding Labels"].apply(
            lambda x: x.split("|")
        )
        self.CLASSES = list(set(chain(*self.df.labels.tolist())))
        self.df.drop(columns=["Finding Labels"], inplace=True)

        # Apply one-hot
        for label in self.CLASSES:
            self.df[label] = self.df.labels.apply(
                lambda x: 1 if label in x else 0
            )

        # Column mapping
        self.df.rename(columns=nih_column_mapping, inplace=True)

        # Full path of filenames
        nih_paths = {
            os.path.basename(x): x
            for x in glob(
                os.path.join(self.dataset_path, "images*", "*", "*.png")
            )
        }
        logger.info(
            f"Images found {len(nih_paths)}. Num rows in df: {self.df.shape[0]}"
        )
        self.df["filepath"] = self.df["filename"].map(nih_paths)

        # Apply original train-test split
        for i, txt in enumerate(["train_val_list.txt", "test_list.txt"]):
            txt = pd.read_table(
                os.path.join(self.dataset_path, txt), header=None
            )
            self.df.loc[self.df.filename.isin(txt[0]), "test_split"] = i
        self.df["test_split"] = self.df["test_split"].astype(int)

        # Create new columns and labels
        self.df["fracture"] = 0
        self.df["nodule_mass"] = 0
        self.df.loc[
            (self.df["nodule"] + self.df["mass"]) > 0, "nodule_mass"
        ] = 1
        self.df["opacity"] = 0
        for opacity_label in OPACITY_LABELS:
            try:
                _opdf += self.df[opacity_label]
            except:
                _opdf = self.df[opacity_label]
        self.df.loc[_opdf > 0, "opacity"] = 1
        self.preprocessing_common()

        # Unify bboxes
        self.df_bbox.rename(columns=nih_column_mapping, inplace=True)
        # Can't be merged as-is because we need to handle several bboxes for the same image
        # self.df_bbox.filename.value_counts(dropna=False) #is more than 1
        # self.df2 = pd.merge(self.df, self.df_bbox, on = 'filename')

        # Drop unused columns
        # self.df.drop(columns = ['image_width', 'image_height', 'image_pixel_spacing_x', 'image_pixel_spacing_y'], inplace = True)
        self.remove_unnamed_columns(self.df)
        self.remove_unnamed_columns(self.df_bbox)

        # Apply expert labels if available
        if use_experts_labels:
            logger.info("Applying expert labels")
            self.integrate_expert_labels("all")


class CheXpertDataset(CXRDataset):
    def __init__(self, dataset_path=CHEXPERT_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "CheXpert"

    def load_dataset(self):
        logger.debug("Importing CheXpert")
        self.orig_train_df = pd.read_csv(
            os.path.join(self.dataset_path, "train.csv")
        )
        self.orig_valid_df = pd.read_csv(
            os.path.join(self.dataset_path, "valid.csv")
        )
        self.orig_test_df = pd.read_csv(
            os.path.join(self.dataset_path, "test.csv")
        )

    def preprocess_dataset(self):
        # Fix Path in test df
        self.orig_test_df["Path"] = (
            "CheXpert-v1.0/" + self.orig_test_df["Path"]
        )

        # Add lateral to test df based on 'lateral' in filename
        # We should base this on the filename, that will end with '_lateral.jpg'
        self.orig_test_df["AP/PA"] = self.orig_test_df["Path"].apply(
            lambda x: "L" if x.endswith("_lateral.jpg") else "PA"
        )

        # Join both datasets
        self.orig_train_df.loc[:, "test_split"] = 0
        self.orig_valid_df.loc[:, "test_split"] = 2
        self.orig_test_df.loc[:, "test_split"] = 1
        self.df = pd.concat(
            [self.orig_train_df, self.orig_valid_df, self.orig_test_df],
            ignore_index=True,
        )
        self.df.loc[:, "dataset"] = self.get_dataset_name()

        # Join projection columns
        self.df.loc[self.df["Frontal/Lateral"] == "Lateral", "AP/PA"] = "L"
        self.df.drop(columns="Frontal/Lateral", inplace=True)

        # Column mapping
        self.df.rename(columns=chexpert_column_mapping, inplace=True)

        # Full path of filenames
        chexpert_paths = lambda x: os.path.join(
            os.path.dirname(DATASETS["CheXpertDataset"]), x
        )
        logger.info(f"Num rows {self.df.shape[0]}")
        self.df["filepath"] = self.df["filename"].map(chexpert_paths)

        # Create patient and study ids
        self.df.loc[:, "patient_id"] = self.df["filename"].apply(
            lambda x: int(x.split("/")[2].strip("patient"))
        )
        self.df.loc[:, "study_id"] = self.df["filename"].apply(
            lambda x: int(x.split("/")[3].strip("study"))
        )
        self.preprocessing_common()

        # Fix patient sex
        self.df.loc[:, "patient_sex"] = self.df["patient_sex"].apply(
            lambda x: x[0] if x == x else np.nan
        )

        # Apply label mapping
        self.df.rename(columns=label_mapping, inplace=True)


class MIMICCXRDataset(CXRDataset):
    def __init__(self, dataset_path=MIMICCXR_PATH):
        super().__init__(dataset_path)
        logger.warning(
            "For now, you need to include metadata, split and chexpert labels from MIMIC-CXR-JPG in the folder"
        )

    @classmethod
    def get_dataset_name(cls):
        return "MIMIC-CXR"

    def add_expert_annotations(self, original_df, test_df):
        # Merge the original dataframe with the test dataframe on 'study_id'
        merged_df = pd.merge(
            original_df,
            test_df,
            on="study_id",
            how="left",
            suffixes=("", "_expert"),
        )

        # List of condition columns to update
        condition_columns = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Lesion",
            "Airspace Opacity",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        # Replace the values in the original columns with the expert annotations
        for column in condition_columns:
            expert_col = column + "_expert"
            if (
                expert_col in merged_df.columns
            ):  # Check if the expert column exists in the merged dataframe
                merged_df[column] = merged_df[expert_col].combine_first(
                    merged_df[column]
                )

        # Drop the expert columns after replacement
        merged_df = merged_df.drop(
            columns=[
                col + "_expert"
                for col in condition_columns
                if col + "_expert" in merged_df.columns
            ]
        )

        return merged_df

    def load_dataset(self, use_negbio_annotations=False):
        logger.debug("Importing MIMIC-CXR")
        self.df1 = pd.read_csv(
            os.path.join(self.dataset_path, "mimic-cxr-2.0.0-metadata.csv.gz"),
            header=0,
            sep=",",
        )
        self.df2 = pd.read_csv(
            os.path.join(self.dataset_path, "mimic-cxr-2.0.0-split.csv.gz"),
            header=0,
            sep=",",
        )
        if use_negbio_annotations:
            self.df3 = pd.read_csv(
                os.path.join(
                    self.dataset_path, "mimic-cxr-2.0.0-negbio.csv.gz"
                ),
                header=0,
                sep=",",
            )
        else:
            self.df3 = pd.read_csv(
                os.path.join(
                    self.dataset_path, "mimic-cxr-2.0.0-chexpert.csv.gz"
                ),
                header=0,
                sep=",",
            )

        self.test_df = pd.read_csv(
            os.path.join(
                self.dataset_path, "mimic-cxr-2.1.0-test-set-labeled.csv"
            )
        )
        merged_outer = pd.merge(
            self.df1,
            self.df2,
            on=["dicom_id", "study_id", "subject_id"],
            how="outer",
        )
        self.df = pd.merge(
            merged_outer, self.df3, on=["study_id", "subject_id"], how="outer"
        )
        self.df = self.add_expert_annotations(self.df, self.test_df)
        self.orig_df = self.df.copy()

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()

        # Full path of filenames
        self.df = self.create_filepath(self.df, base_dir=self.dataset_path)

        # Column mapping
        self.df.rename(columns=mimic_column_mapping, inplace=True)

        # Rename 'LATERAL' and left lateral for 'L' in cxr_projection, leave the rest as is
        self.df.loc[
            self.df["cxr_projection"] == "LATERAL", "cxr_projection"
        ] = "L"
        self.df.loc[self.df["cxr_projection"] == "LL", "cxr_projection"] = "L"

        # Drop procedure_description, study_date, study_time, procedure_code_meaning, view_code_meaning, patient_orientation_code_meaning
        self.df.drop(
            columns=[
                "procedure_description",
                "study_date",
                "study_time",
                "procedure_code_meaning",
                "view_code_meaning",
                "patient_orientation_code_meaning",
            ],
            inplace=True,
        )

        # Combine opacity and airspace opacity, they are one hot encoded so maximum value should be 1 if any if 1
        self.df.loc[:, "opacity"] = self.df.loc[:, "opacity"].combine_first(
            self.df.loc[:, "airspace_opacity"]
        )
        self.df.drop(columns=["airspace_opacity"], inplace=True)

        # Create patient and study ids
        self.preprocessing_common()

        # Apply label mapping
        self.df.rename(columns=label_mapping, inplace=True)

    def create_filepath(self, df, base_dir="files"):
        """
        Create a filepath column for the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing `patient_id` and `study_id` columns.
            base_dir (str): Base directory for the file paths.

        Returns:
            pd.DataFrame: DataFrame with a new `filepath` column.
        """
        # Extract prefix from the patient_id
        df["patient_prefix"] = df["subject_id"].apply(lambda x: str(x)[:2])

        # Construct the filepath
        df["filepath"] = df.apply(
            lambda row: os.path.join(
                base_dir,
                "files",
                f"p{row['patient_prefix']}/",
                f"p{row['subject_id']}/",
                f"s{row['study_id']}/",
                f"{row['dicom_id']}.jpg",
            ),
            axis=1,
        )

        # Drop the intermediate column if desired
        df.drop(columns=["patient_prefix"], inplace=True)

        return df


class PadChestDataset(CXRDataset):
    def __init__(self, dataset_path=PADCHEST_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "PadChest"

    def load_dataset(self):
        logger.debug("Importing PadChest")
        self.orig_df = pd.read_csv(
            os.path.join(
                self.dataset_path,
                "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            )
        )
        self.df = self.orig_df.copy()

    def calculate_patient_age(self, row):
        if (
            row["patient_birth_year"] is not np.nan
            and row["study_date"] is not np.nan
        ):
            return int(str(row["study_date"])[:4]) - row["patient_birth_year"]
        else:
            return np.nan

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()

        # Full path of filenames
        self.df["filepath"] = self.df.apply(
            lambda x: os.path.join(
                self.dataset_path, str(x["ImageDir"]), x["ImageID"]
            ),
            axis=1,
        )

        # Column mapping
        self.df.rename(columns=padchest_column_mapping, inplace=True)

        # Map projection values
        projection_mapping = {
            "PA": "PA",
            "L": "L",
            "AP_horizontal": "AP",
            "AP": "AP",
            "COSTAL": "Other",
            "EXCLUDE": "Other",
            "UNK": "Other",
        }
        self.df.loc[:, "cxr_projection"] = (
            self.df["cxr_projection"].map(projection_mapping).fillna("Other")
        )

        # Create patient_age column from patient_birth_year (int) and studydate (format YYYY)
        self.df.loc[:, "patient_age"] = self.df.apply(
            self.calculate_patient_age, axis=1
        )

        # Create patient and study ids
        self.preprocessing_common()

        # Create labels as list
        self.df["parsed_labels"] = self.df["labels"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

        # Initialize new columns with 0
        self.df["pneumothorax"] = 0
        self.df["normal"] = 0
        self.df["other_disease"] = 0

        # Define condition mappings
        pneumothorax_terms = {
            "pneumothorax",
            " pneumothorax",
            "hydropneumothorax",
            " hydropneumothorax",
        }
        normal_terms = {"normal", " normal"}

        # Update columns based on conditions
        self.df["pneumothorax"] = self.df["parsed_labels"].apply(
            lambda x: 1 if any(term in x for term in pneumothorax_terms) else 0
        )
        self.df["normal"] = self.df["parsed_labels"].apply(
            lambda x: 1 if any(term in x for term in normal_terms) else 0
        )
        self.df["other_disease"] = self.df.apply(
            lambda row: 1
            if row["pneumothorax"] == 0 and row["normal"] == 0
            else 0,
            axis=1,
        )
        self.df["no_finding"] = self.df["normal"]

        # Drop the 'parsed_labels' column
        self.df.drop(columns=["parsed_labels"], inplace=True)

        # Drop columns with _DICOM at the end of the name
        self.df = self.df[
            [col for col in self.df.columns if not col.endswith("_DICOM")]
        ]

        # Remove other columns that are not needed
        unused_columns = [
            "removeit",  # clearly meant to be dropped
            "ImageDir",  # file folder structure — not needed in df
            "patient_birth_year",  # use `patient_age` instead
            "MethodProjection",  # unclear or duplicate of projection
            "ExposureTime",  # raw DICOM field — often not used
            "ReportID",  # only useful if linking to external reports
            "Report",  # free text — not used directly unless NLP
            "MethodLabel",  # may be an alternative label — clarify source
            "Localizations",  # if not using bounding boxes
            "LabelsLocalizationsBySentence",  # free-text label descriptions
            "labelCUIS",  # UMLS — useful for NLP, otherwise not needed
            "LocalizationsCUIS",  # same as above
        ]
        self.df.drop(columns=unused_columns, inplace=True)

        # Apply tube annotations from experts
        self.apply_tube_annotations()

    def apply_tube_annotations(self):
        # Load tube annotations:
        tubes = pd.read_csv(
            os.path.join(
                self.dataset_path,
                "neatx",
                "PadChest_TubeAnnotations_NonExperts_aggregated.csv",
            )
        )

        # There are four columns with tubes: Chest_drain, NSG_tube, Endotracheal_tube, Tracheostomy_tube
        # Make a new column 'support_devices' and set it to 1 if any of the four columns are 1
        tube_cols = [
            "Chest_drain",
            "NSG_tube",
            "Endotracheal_tube",
            "Tracheostomy_tube",
        ]
        tubes["support_devices"] = (
            (tubes[tube_cols] == 1).any(axis=1).astype(int)
        )

        # Now apply the annotations to the df directly. The column to merge both is 'filename' and 'ImageID' in the tubes df
        support_devices_mapping = dict(
            zip(tubes["ImageID"], tubes["support_devices"])
        )
        self.df.loc[
            self.df.filename.isin(support_devices_mapping), "support_devices"
        ] = self.df.filename.map(support_devices_mapping)

        # Drop the tubes df
        del tubes


class VinDrCXRDataset(CXRDataset):
    def __init__(self, dataset_path=VINDRCXR_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "VinDr-CXR"

    def load_dataset(self):
        logger.debug("Importing VinDr-CXR dataset")
        # we dont need to merge the first csv as it contains bbox annotations, we want image level annotations
        self.orig_train_df = pd.read_csv(
            os.path.join(
                self.dataset_path, "annotations", "image_labels_train.csv"
            )
        )
        self.train_df = self.orig_train_df.copy()
        self.orig_test_df = pd.read_csv(
            os.path.join(
                self.dataset_path, "annotations", "image_labels_test.csv"
            )
        )
        self.test_df = self.orig_test_df.copy()

        self.test_df["Other diseases"] = self.test_df["Other disease"]
        self.train_df.loc[:, "test_split"] = 0
        self.test_df.loc[:, "test_split"] = 1
        self.df = pd.concat([self.train_df, self.test_df], ignore_index=True)

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()

        self.df.loc[self.df.test_split == 0, "filepath"] = self.df.apply(
            lambda x: os.path.join(
                self.dataset_path, "train", str(x["image_id"]) + ".dicom"
            ),
            axis=1,
        )
        self.df.loc[self.df.test_split == 1, "filepath"] = self.df.apply(
            lambda x: os.path.join(
                self.dataset_path, "test", str(x["image_id"]) + ".dicom"
            ),
            axis=1,
        )

        # Column mapping
        self.df.rename(columns=vindr_column_mapping, inplace=True)
        # self.collapse_vindr_labels(vindr_column_mapping)

        # Create patient and study ids
        self.df.loc[:, "patient_id"] = self.df.index + 1
        self.df.loc[:, "study_id"] = 1

        # Remove duplicates
        self.df = self.df.drop_duplicates(subset="filename", keep="last")

        self.preprocessing_common()
        self.remove_unnamed_columns(self.df)

        # All images are PA
        self.df["cxr_projection"] = "PA"

    def load_info_from_dicom(self):
        # For each filepath, load the dicom file and get the info
        # Get the patient_id, cxr_projection, patient_sex, patient_age
        # Read the dicom only once
        for filepath in tqdm(
            self.df["filepath_dicom"],
            desc="Loading DICOM info",
            position=0,
            leave=True,
        ):
            dicom = pydicom.dcmread(filepath, stop_before_pixels=True)
            # self.df.loc[self.df['filepath'] == filepath, 'patient_id'] = dicom.PatientID
            try:
                self.df.loc[
                    self.df["filepath_dicom"] == filepath, "cxr_projection"
                ] = dicom.ViewPosition
            except:
                pass
            try:
                self.df.loc[
                    self.df["filepath_dicom"] == filepath, "patient_sex"
                ] = dicom.PatientSex
            except:
                pass
            try:
                self.df.loc[
                    self.df["filepath_dicom"] == filepath, "patient_age"
                ] = int(dicom.PatientAge[:-1])
            except:
                pass

    def load_info_from_dicom_parallel(self, max_workers=8):
        def process_dicom(filepath):
            try:
                dicom = pydicom.dcmread(filepath, stop_before_pixels=True)
                return {
                    "filepath_dicom": filepath,
                    "cxr_projection": getattr(dicom, "ViewPosition", None),
                    "patient_sex": getattr(dicom, "PatientSex", None),
                    "patient_age": int(dicom.PatientAge[:-1])
                    if hasattr(dicom, "PatientAge")
                    and dicom.PatientAge[-1] == "Y"
                    else None,
                }
            except Exception:
                return {
                    "filepath_dicom": filepath,
                    "cxr_projection": None,
                    "patient_sex": None,
                    "patient_age": None,
                }

        filepaths = self.df["filepath_dicom"].tolist()
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_dicom, fp): fp for fp in filepaths
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Reading DICOMs in parallel",
            ):
                results.append(future.result())

        # Merge results into a DataFrame and join with self.df
        dicom_df = pd.DataFrame(results)
        self.df = self.df.merge(dicom_df, on="filepath_dicom", how="left")

    def convert_dicom_to_images(
        self, output_dir: str, output_format: str = "png"
    ):
        logger.debug("Converting DICOM to images")

        # Make a copy of filepath to filepath_dicom
        self.df["filepath_dicom"] = self.df["filepath"]

        # Create progress bar
        total_files = len(self.df)
        pbar = tqdm(
            total=total_files,
            desc="Converting DICOM files",
            position=0,
            leave=True,
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for index, row in self.df.iterrows():
            dicom_file = row["filepath_dicom"]
            # Skip if input and output paths are the same
            output_path = os.path.join(
                output_dir, os.path.basename(dicom_file) + f".{output_format}"
            )
            if dicom_file == output_path:
                pbar.update(1)
                continue

            # Update filepath to point to the new png image
            self.df.at[index, "filepath"] = output_path

            # Read DICOM file
            dicom = pydicom.dcmread(dicom_file)

            # Get correct pixel data accounting for photometric interpretation
            if hasattr(dicom, "PhotometricInterpretation"):
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    image = np.max(dicom.pixel_array) - dicom.pixel_array
                else:
                    image = dicom.pixel_array
            else:
                image = dicom.pixel_array

            # Apply windowing if available
            if hasattr(dicom, "WindowCenter") and hasattr(
                dicom, "WindowWidth"
            ):
                center = dicom.WindowCenter
                width = dicom.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]

                # Apply windowing formula
                vmin = center - width / 2
                vmax = center + width / 2
                image = np.clip(image, vmin, vmax)

            # Rescale to 16-bit
            if hasattr(dicom, "RescaleIntercept") and hasattr(
                dicom, "RescaleSlope"
            ):
                image = image * dicom.RescaleSlope + dicom.RescaleIntercept

            # Normalize to 0-255 range
            image = (
                (image - image.min()) / (image.max() - image.min()) * 255
            ).astype(np.uint8)

            # Convert to PIL and save
            image = Image.fromarray(image)
            image.save(output_path)

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

    def collapse_vindr_labels(self):
        from collections import defaultdict

        # Reverse mapping: final column name → list of original columns
        grouped_columns = defaultdict(list)
        for original_col, target_col in vindr_column_mapping.items():
            if original_col != "image_id":
                grouped_columns[target_col].append(original_col)

        df = self.df.copy()

        # Create final merged columns
        for new_col, original_cols in grouped_columns.items():
            # Filter only columns that exist in the df
            existing_cols = [col for col in original_cols if col in df.columns]
            if existing_cols:
                df[new_col] = (df[existing_cols] == 1).any(axis=1).astype(int)

        # Rename image_id to filename if necessary
        if "image_id" in df.columns:
            df = df.rename(columns={"image_id": "filename"})

        self.df = df


class BRAXDataset(CXRDataset):
    def __init__(self, dataset_path=BRAX_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "BRAX"

    def load_dataset(self):
        logger.debug("Importing brax")
        self.orig_df = pd.read_csv(
            os.path.join(self.dataset_path, "master_spreadsheet_update.csv")
        )
        self.df = self.orig_df.copy()

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()

        # Full path of filenames
        self.df["filepath"] = self.df.apply(
            lambda x: os.path.join(self.dataset_path, x["PngPath"]), axis=1
        )
        self.df["filepath_dicom"] = self.df.apply(
            lambda x: os.path.join(self.dataset_path, x["DicomPath"]), axis=1
        )

        # Column mapping
        self.df.rename(columns=brax_column_mapping, inplace=True)

        # Create patient and study ids
        self.preprocessing_common()


class PediatricPneumoniaDataset(CXRDataset):
    def __init__(self, dataset_path=PEDIATRICPNEUMONIA_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "Pediatric_Pneumonia"

    def load_dataset(self):
        logger.debug("Importing GuanzhouPediatricPneumonia...")
        folders = ["train", "val", "test"]
        labels = ["NORMAL", "PNEUMONIA"]
        combinations = [(a, b) for a in folders for b in labels]


class JSRTDataset(CXRDataset):
    def __init__(self, dataset_path=JSRT_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "JSRT"

    def convert_single_image(self, filepath: str):
        logging.info(f"Converting {filepath}")
        try:
            shape = (2048, 2048)
            dtype = np.dtype(">u2")  # big-endian unsigned integer (16-bit)
            output_filename = filepath.replace(".IMG", ".png")
            with open(filepath, "rb") as f:
                data = np.fromfile(f, dtype=dtype)
                image = data.reshape(shape)
                # Normalize to 0-255 range and convert to uint8
                # image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                # Normalization mode 2
                low, high = np.percentile(image, (1, 99))
                image = np.clip(image, low, high)
                image = ((image - low) / (high - low) * 255).astype(np.uint8)
                # Make it negative
                image = 255 - image
                Image.fromarray(image).save(output_filename)
        except Exception as e:
            logging.error(f"Error converting {filepath}: {e}")
            return np.nan
        return output_filename

    def convert_dicom_to_images(self):
        self.df["filepath_dicom"] = self.df["filepath"]
        self.df["filepath"] = self.df["filepath"].apply(
            self.convert_single_image
        )

    def load_dataset(self):
        logger.debug("Importing JSRT Dataset")
        self.df1 = pd.read_table(
            os.path.join(
                self.dataset_path, "Clinical_Information", "CLNDAT_EN.txt"
            ),
            header=None,
            sep="\t",
        )
        self.df2 = pd.read_table(
            os.path.join(
                self.dataset_path, "Clinical_Information", "CNNDAT_EN.TXT"
            ),
            header=None,
            sep="\s+",
        )

    def preprocess_dataset(self):
        # Join both datasets
        df1_columns = [
            "filename",
            "subtlety",
            "size",
            "patient_age",
            "patient_sex",
            "bbox_x",
            "bbox_y",
            "diagnosis",
            "location",
            "report",
        ]
        # Fix data displaying in two columns
        self.df1[8] = self.df1[8] + " " + self.df1[9].fillna("")
        self.df1[10] = self.df1[10] + self.df1[11].fillna("")
        self.df1 = self.df1.drop(columns=[9, 11])
        self.df1.columns = df1_columns
        self.df1["no_finding"] = 0
        self.df1["nodule_mass"] = 1

        df2_columns = [
            "filename",
            "patient_age",
            "patient_gender",
            "diagnosis",
        ]
        self.df2.columns = df2_columns
        self.df2["no_finding"] = 1
        self.df2["nodule_mass"] = 0
        self.df = pd.concat([self.df1, self.df2], ignore_index=True)

        # Convert patient_sex to F/M
        self.df["patient_sex"] = self.df["patient_sex"].apply(
            lambda x: "F" if x == "female" else "M"
        )

        # Full path of filenames
        jsrt_paths = lambda x: os.path.join(
            DATASETS["JSRTDataset"], "All247images", x
        )
        logger.info(f"Num rows: {self.df.shape[0]}")
        self.df["filepath"] = self.df["filename"].map(jsrt_paths)

        # Create patient and study ids
        # Assign unique id to each patient:
        self.df.loc[:, "patient_id"] = self.df.index + 1
        self.df.loc[:, "study_id"] = 1
        self.preprocessing_common()

        # Add dataset name
        self.df["dataset"] = self.get_dataset_name()

        # Add cxr_projection and other labels
        self.df["cxr_projection"] = "PA"
        self.df["pneumothorax"] = 0
        self.df["support_devices"] = 0
        self.df["test_split"] = 0


class SIIMACRPneumothoraxDataset(CXRDataset):
    def __init__(self, dataset_path=SIIM_PNEUMOTHORAX_PATH):
        super().__init__(dataset_path)
        self.column_mapping = {
            "ImageId": "filename",
        }

    @classmethod
    def get_dataset_name(cls):
        return "SIIM-ACR"

    def _construct_filepath_siim(self, row, root_folder):
        subfolder = "test_png" if row["test"] == 1 else "train_png"
        return f"{root_folder}/{subfolder}/{row['ImageId']}.png"

    def create_csv(self):
        training_path = self.dataset_path + "/train_png/*.png"
        test_path = self.dataset_path + "/test_png/*.png"
        training_files = glob(training_path)
        test_files = glob(test_path)
        train_filenames = [Path(elem).stem for elem in training_files]
        test_filenames = [Path(elem).stem for elem in test_files]
        metadata = pd.read_csv(
            self.dataset_path + "/train-rle.csv", index_col=None
        )
        df["pneumothorax"] = (df[" EncodedPixels"] != "-1").astype(int)
        df["patient_id"] = df["filepath"].apply(
            lambda x: int(x.split("/")[-1].split(".")[0])
        )
        df["test_split"] = df["ImageId"].apply(
            lambda x: 1
            if x in test_filenames
            else (0 if x in train_filenames else -1)
        )
        df["filepath"] = df.apply(
            lambda row: self._construct_filepath_siim(row, self.dataset_path),
            axis=1,
        )
        # df = df.drop_duplicates(subset=['ImageId'], keep=False)
        return df

    def load_dataset(self):
        self.orig_df = pd.read_csv(
            "segmentation/pneumothorax_siim_clasification.csv"
        )
        self.df = self.orig_df.copy()

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()
        self.df.rename(columns=self.column_mapping, inplace=True)
        self.df = self.df.drop_duplicates(subset=["filename"], keep="first")
        # Drop EncodedPixels column
        self.df = self.df.drop(columns=[" EncodedPixels"])
        # Create patient and study ids
        self.df.loc[:, "patient_id"] = self.df.index + 1
        self.df.loc[:, "study_id"] = 1
        self.preprocessing_common()
        # Create no_finding column if no pneumothorax is found
        self.df["no_finding"] = self.df["pneumothorax"].apply(
            lambda x: 1 if x == 0 else 0
        )
        # Convert filesize to image_width and image_height
        import ast

        self.df["filesize"] = self.df["filesize"].apply(
            lambda x: ast.literal_eval(x)
        )
        self.df["image_width"] = self.df["filesize"].apply(lambda x: int(x[0]))
        self.df["image_height"] = self.df["filesize"].apply(
            lambda x: int(x[1])
        )

        # All images are PA
        self.df["cxr_projection"] = "PA"


class PTX498Dataset(CXRDataset):
    def __init__(self, dataset_path=PTX498_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "PTX498"

    def create_csv(self):
        logger.debug("Creating PTX498 Dataset CSV")
        sites = ["SiteA", "SiteB", "SiteC"]
        image_files = []

        for site in sites:
            site_path = os.path.join(self.dataset_path, site)
            if os.path.exists(site_path):
                # Get only the image files, not the mask files
                files = glob(os.path.join(site_path, "*.img.png"))
                site_name = site[-1]
                image_files.extend([(f, site_name) for f in files])

        # Create DataFrame
        df = pd.DataFrame(image_files, columns=["filepath", "site"])

        # Extract filename from filepath
        df["filename"] = df["filepath"].apply(lambda x: os.path.basename(x))

        # All images in this dataset have pneumothorax
        df["pneumothorax"] = 1
        df["test_split"] = 0  # All in training set for now

        # Create patient IDs with site prefix
        # df['patient_id'] = df.apply(lambda x: f"{x['site'][-1]}{x.name + 1}", axis=1)
        df["patient_id"] = df.index + 1

        # Save CSV
        csv_path = os.path.join(self.dataset_path, "ptx498_metadata.csv")
        df.to_csv(csv_path, index=False)
        return df

    def load_dataset(self):
        logger.debug("Importing PTX498 Dataset")
        csv_path = os.path.join(self.dataset_path, "ptx498_metadata.csv")

        if not os.path.exists(csv_path):
            logger.info("CSV not found. Creating new one...")
            self.df = self.create_csv()
        else:
            self.df = pd.read_csv(csv_path)

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()
        self.df.loc[:, "study_id"] = 1
        self.df["no_finding"] = 0

        self.preprocessing_common()
        self.remove_unnamed_columns(self.df)

        # All images are PA
        self.df["cxr_projection"] = "PA"


class CANDIDPTXDataset(CXRDataset):
    def __init__(self, dataset_path=CANDIDPTX_PATH):
        super().__init__(dataset_path)
        self.column_mapping = {
            "SOPInstanceUID": "image_id",
            "StudyInstanceUID": "study_id",
            "EncodedPixels": "segmentation_mask",
            "Report": "report",
        }

    @classmethod
    def get_dataset_name(cls):
        return "CANDID-PTX"

    def create_csv(self):
        logger.debug("Creating CANDID-PTX Dataset CSV")

        # Load base csv
        df = pd.read_csv(
            os.path.join(self.dataset_path, "Pneumothorax_reports.csv")
        )

        # Add filepath column
        df["filepath"] = df["SOPInstanceUID"].apply(
            lambda x: os.path.join(self.dataset_path, "dataset", x)
        )

        # Get the patient_id from the filename (PXXX_YYYYMMDD_HHMMSS)
        df["patient_id"] = df["filename"].apply(
            lambda x: int(x.split("_")[0].replace("P", ""))
        )

        # Get study_date from filename after patient_id (DD_MM_YYYY) and before _DICOM_ (eg: from P0_04_11_2563_DICOM_ is 04_11_2563)
        # We must extract it between the first and the fourth underscore, and then keep the years between 2010-2019
        df["study_date"] = (
            df["filename"].apply(lambda x: "/".join(x.split("_")[1:3]))
            + "/"
            + df["filename"].apply(lambda x: str(int(x.split("_")[3]) - 553))
        )

        # Add pneumothorax if segmentation mask is not "-1"
        df["pneumothorax"] = df["EncodedPixels"].apply(
            lambda x: 0 if x == "-1" else 1
        )
        df["no_finding"] = df["pneumothorax"].apply(
            lambda x: 1 if x == 0 else 0
        )

        # Add standard columns
        df["test_split"] = 0
        df["support_devices"] = 0
        df["fracture"] = 0
        # Add support devices
        devices = pd.read_csv(
            os.path.join(self.dataset_path, "chest_tube.csv")
        )
        devices_list = devices.anon_SOPUID.tolist()
        df.loc[df.SOPInstanceUID.isin(devices_list), "support_devices"] = 1

        # Add rib fractures
        rib_fractures = pd.read_csv(
            os.path.join(self.dataset_path, "Rib_fracture_mask_rle.csv")
        )
        rib_fractures_list = rib_fractures.anon_SOPUID.tolist()
        df.loc[df.SOPInstanceUID.isin(rib_fractures_list), "fracture"] = 1

        # Remove duplicated SOPInstanceUID rows (there are duplicated masks)
        df = df.drop_duplicates(subset=["SOPInstanceUID"], keep="first")

        # Save CSV
        csv_path = os.path.join(self.dataset_path, "candid_ptx_metadata.csv")
        df.to_csv(csv_path, index=False)
        return df

    def load_dataset(self):
        logger.debug("Importing CANDID-PTX Dataset")
        csv_path = os.path.join(self.dataset_path, "candid_ptx_metadata.csv")

        if not os.path.exists(csv_path):
            logger.info("CSV not found. Creating new one...")
            self.df = self.create_csv()
        else:
            self.df = pd.read_csv(csv_path)

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()
        self.df.rename(columns=self.column_mapping, inplace=True)
        self.preprocessing_common()
        self.remove_unnamed_columns(self.df)

        # Remove report and segmentation mask columns
        if (
            "report" in self.df.columns
            and "segmentation_mask" in self.df.columns
        ):
            self.df = self.df.drop(columns=["report", "segmentation_mask"])

        # All images are PA
        self.df["cxr_projection"] = "PA"

    def convert_dicom_to_images(
        self, output_dir: str, output_format: str = "png"
    ):
        logger.debug("Converting DICOM to images")

        # Make a copy of filepath to filepath_dicom
        self.df["filepath_dicom"] = self.df["filepath"]

        # Create progress bar
        total_files = len(self.df)
        pbar = tqdm(
            total=total_files,
            desc="Converting DICOM files",
            position=0,
            leave=True,
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for index, row in self.df.iterrows():
            dicom_file = row["filepath_dicom"]
            # Skip if input and output paths are the same
            output_path = os.path.join(
                output_dir, os.path.basename(dicom_file) + f".{output_format}"
            )
            if dicom_file == output_path:
                pbar.update(1)
                continue

            # Update filepath to point to the new png image
            self.df.at[index, "filepath"] = output_path

            # Read DICOM file
            dicom = pydicom.dcmread(dicom_file)

            # Get correct pixel data accounting for photometric interpretation
            if hasattr(dicom, "PhotometricInterpretation"):
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    image = np.max(dicom.pixel_array) - dicom.pixel_array
                else:
                    image = dicom.pixel_array
            else:
                image = dicom.pixel_array

            # Apply windowing if available
            if hasattr(dicom, "WindowCenter") and hasattr(
                dicom, "WindowWidth"
            ):
                center = dicom.WindowCenter
                width = dicom.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]

                # Apply windowing formula
                vmin = center - width / 2
                vmax = center + width / 2
                image = np.clip(image, vmin, vmax)

            # Rescale to 16-bit
            if hasattr(dicom, "RescaleIntercept") and hasattr(
                dicom, "RescaleSlope"
            ):
                image = image * dicom.RescaleSlope + dicom.RescaleIntercept

            # Normalize to 0-255 range
            image = (
                (image - image.min()) / (image.max() - image.min()) * 255
            ).astype(np.uint8)

            # Convert to PIL and save
            image = Image.fromarray(image)
            image.save(output_path)

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()


class CRADIDataset(CXRDataset):
    def __init__(self, dataset_path=CRADI_PATH):
        super().__init__(dataset_path)
        self.dataset_labels = [
            "support_devices",
            "cardiomegaly",
            "consolidation",
            "emphysema",
            "mass",
            "pleural_other",
            "pleural_effusion",
            "pleural_thickening",
            "pneumothorax",
            "edema",
            "nodule",
            "other",
        ]

    @classmethod
    def get_dataset_name(cls):
        return "CRADI"

    def load_dataset(self):
        logger.debug("Importing CRADI Dataset")
        # Read the Excel file with labels
        self.df = pd.read_excel(
            os.path.join(self.dataset_path, "CHARDI_label.xlsx")
        )

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()

        # Create filepath using the test directory and filename
        self.df["filepath"] = self.df["new_id"].apply(
            lambda x: os.path.join(
                self.dataset_path, "CHARDI_test", x + ".png"
            )
        )

        # Create test_split column
        self.df["test_split"] = 0
        test_files = glob(
            os.path.join(self.dataset_path, "CHARDI_test", "*.png")
        )
        test_filenames = [Path(elem).stem for elem in test_files]
        self.df["test_split"] = self.df["new_id"].apply(
            lambda x: 1 if x in test_filenames else 0
        )

        # Create patient_id using new_id column
        self.df.loc[:, "patient_id"] = self.df.index + 1
        self.df.loc[:, "study_id"] = 1  # Single study per patient
        self.df.loc[:, "image_id"] = self.df["new_id"].apply(
            lambda x: int(x.split("image")[1])
        )

        # Combine similar findings (consolidation types)
        consolidation_cols = [
            "consolidation",
            "patchy consolidation",
            "small consolidation",
        ]
        self.df["consolidation"] = self.df[consolidation_cols].max(axis=1)

        # Combine support devices
        device_cols = ["PICC implant", "pacemaker implant"]
        self.df["support_devices"] = self.df[device_cols].max(axis=1)

        # Combine aortic abnormalities
        aortic_cols = [
            "aortic abnormalities",
            "aortic arteriosclerosis",
            "aortic unfolding",
        ]
        self.df["aortic_abnormalities"] = self.df[aortic_cols].max(axis=1)

        # Combine pleural abnormalities
        pleural_cols = [
            "pleural abnormalities",
            "pleural adhesion",
            "pleural calcification",
        ]
        self.df["pleural_abnormalities"] = self.df[pleural_cols].max(axis=1)

        # Rename columns according to mapping
        self.df.drop(
            columns=[
                "PICC implant",
                "pacemaker implant",
                "patchy consolidation",
                "small consolidation",
                "aortic abnormalities",
                "aortic arteriosclerosis",
                "aortic unfolding",
                "pleural abnormalities",
                "pleural adhesion",
                "pleural calcification",
            ],
            inplace=True,
        )
        self.df.rename(columns=cradi_column_mapping, inplace=True)

        # Calculate no_finding (1 if all other findings are 0)
        finding_cols = [
            col for col in self.df.columns if col in self.dataset_labels
        ]
        self.df["no_finding"] = (
            self.df[finding_cols].sum(axis=1) == 0
        ).astype(int)

        self.preprocessing_common()
        self.remove_unnamed_columns(self.df)

        # Remove all elements not present yet (all that are not 1 in test_split)
        self.df = self.df[self.df["test_split"] == 1]

        # All images are PA
        self.df["cxr_projection"] = "PA"


class BelarusDataset(CXRDataset):
    def __init__(self, dataset_path=BELARUS_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "Belarus"

    def create_csv(self):
        logger.debug("Creating Belarus Dataset CSV")
        df = pd.DataFrame()
        files = [
            file
            for file in os.listdir(os.path.join(self.dataset_path))
            if file.endswith(".png")
        ]
        df["filename"] = files
        self.df = df

    def load_dataset(self):
        logger.debug("Importing Belarus Dataset")
        csv_path = os.path.join(self.dataset_path, "dataset.csv")
        if not os.path.exists(csv_path):
            logger.info("CSV not found. Creating new one...")
            self.create_csv()
        else:
            self.df = pd.read_csv(csv_path)

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()
        self.df["filepath"] = self.df["filename"].apply(
            lambda x: os.path.join(self.dataset_path, x)
        )
        self.df["patient_id"] = self.df.index + 1
        self.df["image_id"] = self.df.index + 1
        self.df["study_id"] = 1
        self.df["test_split"] = 0
        self.df["no_finding"] = 0
        self.df["pneumothorax"] = 0
        self.df["TBC"] = 1
        self.df["support_devices"] = 0
        self.df["cxr_projection"] = "PA"
        self.preprocessing_common()


class ObjectCXRDataset(CXRDataset):
    def __init__(self, dataset_path=OBJECT_CXR_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "Object-CXR"

    def load_dataset(self):
        self.orig_train = pd.read_csv(
            os.path.join(self.dataset_path, "train.csv")
        )
        self.orig_dev = pd.read_csv(os.path.join(self.dataset_path, "dev.csv"))
        # self.orig_test = pd.read_csv(os.path.join(self.dataset_path, 'test.csv'))

    def preprocess_dataset(self):
        # Create filepath and split for train
        self.orig_train.rename(
            columns={"image_name": "filename"}, inplace=True
        )
        self.orig_train["filepath"] = self.orig_train["filename"].apply(
            lambda x: os.path.join(self.dataset_path, "train", x)
        )
        self.orig_train["test_split"] = 0

        # Create filepath and split for dev
        self.orig_dev.rename(columns={"image_name": "filename"}, inplace=True)
        self.orig_dev["filepath"] = self.orig_dev["filename"].apply(
            lambda x: os.path.join(self.dataset_path, "dev", x)
        )
        self.orig_dev["test_split"] = 2

        # Create filepath and split for dev
        # self.orig_test.rename(columns={'image_name': 'filename'}, inplace=True)
        # self.orig_test['filepath'] = self.orig_test['filename'].apply(lambda x: os.path.join(self.dataset_path, 'test', x))
        # self.orig_test['test_split'] = 1

        # Combine both datasets
        self.df = pd.concat([self.orig_train, self.orig_dev])

        # Create patient_id, study_id and image_id for train and dev
        self.df["patient_id"] = self.df.index + 1
        self.df["study_id"] = 1
        self.df["image_id"] = self.df.index + 1
        self.df["dataset"] = self.get_dataset_name()

        # Create no_finding column and set to 1 if annotation column is na
        # Do the opposite for foreign_object column
        self.df["no_finding"] = 0
        self.df.loc[self.df["annotation"].isna(), "no_finding"] = 1
        self.df["foreign_object"] = 0
        self.df.loc[self.df["annotation"].notna(), "foreign_object"] = 1

        # Create cxr_projection column
        self.df["cxr_projection"] = "PA"

        # Create support_devices and pneumothoraxcolumn
        self.df["support_devices"] = 0
        self.df["pneumothorax"] = 0

        # Create dataset column
        self.df["dataset"] = self.get_dataset_name()

        # Drop annotation
        self.df.drop(columns=["annotation"], inplace=True)

        # Preprocess common
        self.preprocessing_common()


class MontgomeryDataset(CXRDataset):
    def __init__(self, dataset_path=MONTGOMERY_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "Montgomery-TBC"

    def create_csv(self):
        logger.debug("Creating Montgomery-TBC Dataset CSV")
        df = pd.DataFrame()
        files = [
            file
            for file in os.listdir(os.path.join(self.dataset_path, "CXR_png"))
            if file.endswith(".png")
        ]
        df["filename"] = files
        self.df = df

    def load_dataset(self):
        logger.debug("Importing Montgomery-TBC Dataset")
        csv_path = os.path.join(self.dataset_path, "dataset.csv")
        if not os.path.exists(csv_path):
            logger.info("CSV not found. Creating new one...")
            self.create_csv()
        else:
            self.df = pd.read_csv(csv_path)

    def read_clinical_info(self):
        # Append infor from ClinicalReadings folder to dataframe
        # Each filename has a corresponding txt file in the ClinicalReadings folder
        # Read the txt file and append the information to the dataframe
        for index, row in self.df.iterrows():
            txt_path = os.path.join(
                self.dataset_path,
                "ClinicalReadings",
                row["filename"].replace(".png", ".txt"),
            )
            with open(txt_path, "r") as file:
                lines = file.readlines()
                # First line is patient_sex, second line is patient_age, third is report
                self.df.at[index, "patient_sex"] = lines[0].strip()
                self.df.at[index, "patient_age"] = lines[1].strip()
                self.df.at[index, "report"] = lines[2].strip()

        # Remove "Patient's Sex: " from patient_sex column
        self.df["patient_sex"] = (
            self.df["patient_sex"]
            .str.replace("Patient's Sex: ", "")
            .str[0]
            .str.upper()
        )

        # Remove "Patient's Age: " from patient_age column. Also remove the last character (89Y)
        self.df["patient_age"] = (
            self.df["patient_age"]
            .str.replace("Patient's Age: ", "")
            .str[:-1]
            .astype(int)
        )

    def preprocess_dataset(self):
        self.df.loc[:, "dataset"] = self.get_dataset_name()
        self.df["filepath"] = self.df["filename"].apply(
            lambda x: os.path.join(self.dataset_path, "CXR_png", x)
        )

        # Read clinical info
        self.read_clinical_info()

        # Create patient_id, study_id and image_id
        self.df["patient_id"] = self.df.index + 1
        self.df["study_id"] = 1
        self.df["image_id"] = self.df.index + 1

        # Create no_finding column and set to 1 if report is empty
        self.df["no_finding"] = 0
        self.df["TBC"] = 1
        self.df.loc[
            (self.df["report"] == "normal") | (self.df["report"] == ""),
            "no_finding",
        ] = 1
        self.df.loc[
            (self.df["report"] == "normal") | (self.df["report"] == ""), "TBC"
        ] = 0

        self.df["pneumothorax"] = 0
        self.df["support_devices"] = 0

        # Create cxr_projection column
        self.df["cxr_projection"] = "PA"

        # Preprocess common
        self.preprocessing_common()

        # Remove report column
        self.df.drop(columns=["report"], inplace=True)


class ShenzhenDataset(MontgomeryDataset):
    def __init__(self, dataset_path=SHENZHEN_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "Shenzhen-TBC"

    def read_clinical_info(self):
        # Append infor from ClinicalReadings folder to dataframe
        # Each filename has a corresponding txt file in the ClinicalReadings folder
        # Read the txt file and append the information to the dataframe
        for index, row in self.df.iterrows():
            txt_path = os.path.join(
                self.dataset_path,
                "ClinicalReadings",
                row["filename"].replace(".png", ".txt"),
            )
            with open(txt_path, "r") as file:
                lines = file.readlines()
                # First line is patient_sex, second line is patient_age, third is report
                self.df.at[index, "patient_sex"] = lines[0].strip()
                self.df.at[index, "patient_age"] = lines[0].strip()
                self.df.at[index, "report"] = lines[1].strip()

        # Get first digit from patient_sex and convert to uppercase
        self.df["patient_sex"] = self.df["patient_sex"].str[0].str.upper()

        # Get the digits from the patient_age column using regex
        self.df["patient_age"] = (
            self.df["patient_age"].str.extract(r"(\d+)").astype(int)
        )
        # self.df['patient_age'] = self.df['patient_age'].str[-5:-3]


class TBX11KDataset(CXRDataset):
    def __init__(self, dataset_path=TBX11K_PATH):
        super().__init__(dataset_path)
        self.dataset_labels = [
            "Healthy",
            "Sick but Non-TB",
            "Active TB",
            "Latent TB",
            "Uncertain TB",
            "tuberculosis",
            "no_finding",
        ]  # Added placeholder 'tuberculosis' and 'no_finding'

    @classmethod
    def get_dataset_name(cls):
        return "TBX11K"

    def _load_split(self, split_name: str) -> List[str]:
        """Loads image IDs from a split file."""
        split_file = os.path.join(
            self.dataset_path, "lists", f"{split_name}.txt"
        )
        try:
            with open(split_file, "r") as f:
                image_ids = [line.strip() for line in f.readlines()]
            return image_ids
        except FileNotFoundError:
            logger.error(f"Split file not found: {split_file}")
            return []

    def load_dataset(self):
        logger.debug(f"Importing {self.get_dataset_name()} dataset splits")
        train_ids = self._load_split("all_train")
        val_ids = self._load_split("all_val")
        test_ids = self._load_split("all_test")

        # Combine IDs and create DataFrame
        all_ids = train_ids + val_ids + test_ids
        if not all_ids:
            raise FileNotFoundError("No split files found for TBX11K dataset.")

        self.df = pd.DataFrame({"filename": all_ids})

        # Assign splits
        self.df["test_split"] = -1  # Default value
        self.df.loc[self.df["filename"].isin(train_ids), "test_split"] = (
            0  # 0 for train
        )
        self.df.loc[self.df["filename"].isin(test_ids), "test_split"] = (
            1  # 1 for test
        )
        self.df.loc[self.df["filename"].isin(val_ids), "test_split"] = (
            2  # 2 for validation
        )

        # Remove info from other datasets:
        self.df = self.df.loc[~self.df.filename.str.startswith("extra")]

        logger.info(f"Num rows: {len(self.df)}")

    def preprocess_dataset(self):
        # Get label from filename:
        # health are no_finding 1 and else 0; tb is TBC 1 and no_finding 0; sick is TBC 0 and no_finding 0
        self.df["no_finding"] = 0
        self.df["TBC"] = 0
        self.df["pneumothorax"] = 0
        self.df["support_devices"] = 0
        self.df.loc[
            self.df.filename.str.startswith("health"), "no_finding"
        ] = 1
        self.df.loc[self.df.filename.str.startswith("tb"), "TBC"] = 1

        # remove labels from test split as we don't know them
        self.df.loc[self.df.test_split == 1, "no_finding"] = np.nan
        self.df.loc[self.df.test_split == 1, "TBC"] = np.nan

        self.df["dataset"] = self.get_dataset_name()

        # Construct filepath - Assuming images are in 'imgs/' directory and are png
        self.df["filepath"] = self.df["filename"].apply(
            lambda x: os.path.join(self.dataset_path, "imgs", x)
        )

        # Placeholder for patient/study ID - using image ID for now
        self.df["patient_id"] = self.df.index + 1
        self.df["study_id"] = 1

        # Common preprocessing steps
        self.preprocessing_common()

        # Add dataset-specific info
        self.df["image_width"] = 512
        self.df["image_height"] = 512
        self.df["cxr_projection"] = "PA"

        self.remove_unnamed_columns(self.df)


class DADBDataset(CXRDataset):
    def __init__(self, dataset_path=DADB_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "DA-DB"

    def load_dataset(self):
        logger.debug("Importing DA-DB Dataset")
        # Get files from folders
        da_path = os.path.join(self.dataset_path, "images", "da")
        db_path = os.path.join(self.dataset_path, "images", "db")
        da_images = [
            file for file in os.listdir(da_path) if file.endswith(".jpg")
        ]
        db_images = [
            file for file in os.listdir(db_path) if file.endswith(".dcm")
        ]
        da_fullpath = [os.path.join(da_path, image) for image in da_images]
        db_fullpath = [os.path.join(db_path, image) for image in db_images]
        self.df = pd.DataFrame(
            {
                "filename": da_images + db_images,
                "filepath": da_fullpath + db_fullpath,
            }
        )
        # Get source from filename
        self.df["source"] = "DA"
        self.df.loc[self.df.filename.isin(db_images), "source"] = "DB"

    def convert_dicom_to_images(
        self, output_dir: str, output_format: str = "png"
    ):
        logger.debug("Converting DICOM to images")

        # Get DICOM filepath for DB
        self.df.loc[self.df.source == "DB", "filepath_dicom"] = self.df.loc[
            self.df.source == "DB", "filepath"
        ]

        # Create progress bar
        total_files = len(self.df.loc[self.df.source == "DB"])
        pbar = tqdm(
            total=total_files,
            desc="Converting DICOM files",
            position=0,
            leave=True,
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for index, row in self.df.loc[self.df.source == "DB"].iterrows():
            dicom_file = row["filepath_dicom"]
            # Skip if input and output paths are the same
            output_path = os.path.join(
                output_dir, os.path.basename(dicom_file) + f".{output_format}"
            )
            if dicom_file == output_path:
                pbar.update(1)
                continue

            # Update filepath to point to the new png image
            self.df.at[index, "filepath"] = output_path

            # Read DICOM file
            dicom = pydicom.dcmread(dicom_file)

            # Get correct pixel data accounting for photometric interpretation
            if hasattr(dicom, "PhotometricInterpretation"):
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    image = np.max(dicom.pixel_array) - dicom.pixel_array
                else:
                    image = dicom.pixel_array
            else:
                image = dicom.pixel_array

            # Apply windowing if available
            if hasattr(dicom, "WindowCenter") and hasattr(
                dicom, "WindowWidth"
            ):
                center = dicom.WindowCenter
                width = dicom.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]

                # Apply windowing formula
                vmin = center - width / 2
                vmax = center + width / 2
                image = np.clip(image, vmin, vmax)

            # Rescale to 16-bit
            if hasattr(dicom, "RescaleIntercept") and hasattr(
                dicom, "RescaleSlope"
            ):
                image = image * dicom.RescaleSlope + dicom.RescaleIntercept

            # Normalize to 0-255 range
            image = (
                (image - image.min()) / (image.max() - image.min()) * 255
            ).astype(np.uint8)

            # Convert to PIL and save
            image = Image.fromarray(image)
            image.save(output_path)

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

    def preprocess_dataset(self):
        # Get split from filename
        self.df["test_split"] = -1
        self.df.loc[self.df.filename.str.contains("train"), "test_split"] = 0
        self.df.loc[self.df.filename.str.contains("test"), "test_split"] = 1

        # Create labels
        self.df["no_finding"] = 0
        self.df["TBC"] = 0
        self.df["pneumothorax"] = 0
        self.df["support_devices"] = 0

        # Add label from filename: p is for positive and should add TBC; n is for negative and should add no_finding
        self.df.loc[self.df.filename.str.contains("_p"), "TBC"] = 1
        self.df.loc[self.df.filename.str.contains("_n"), "no_finding"] = 1

        # Add dataset name
        self.df["dataset"] = self.get_dataset_name()

        # Create patient_id, study_id and image_id
        self.df["patient_id"] = self.df.index + 1
        self.df["study_id"] = 1
        self.df["image_id"] = self.df.index + 1

        # Add dataset name
        self.df["dataset"] = self.get_dataset_name()

        # Common preprocessing steps
        self.preprocessing_common()

        # Create cxr_projection column
        self.df["cxr_projection"] = "PA"

        # Remove filepath_dicom column
        # self.df.drop(columns=['filepath_dicom'], inplace=True)


class RANZCRDataset(CXRDataset):
    def __init__(self, dataset_path=RANZCR_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "RANZCR-CLIP"

    def load_dataset(self):
        logger.debug("Importing RANZCR-CLIP Dataset")
        # Get files from folders
        self.df = pd.read_csv(os.path.join(self.dataset_path, "train.csv"))

        logging.warning("Not loading images from test set for now")
        self.df["test_split"] = 0

    def preprocess_dataset(self):
        # Rename columns
        self.df.rename(
            columns={
                "PatientID": "patient_id",
                "StudyInstanceUID": "image_id",
            },
            inplace=True,
        )

        # Create study_id
        self.df["study_id"] = 1

        # Create filename
        self.df["filename"] = self.df["image_id"].astype(str) + ".jpg"
        self.df["filepath"] = self.df["filename"].apply(
            lambda x: os.path.join(self.dataset_path, "train", x)
        )

        # Create labels
        self.df["support_devices"] = 1
        self.df["no_finding"] = (
            0  # a quick look of the images show that most contain pathology, but we don't know what
        )

        # Add dataset name
        self.df["dataset"] = self.get_dataset_name()

        # Common preprocessing steps
        self.preprocessing_common()

        # Create cxr_projection column
        self.df["cxr_projection"] = "PA"

        # Remove other columns:
        columns_to_remove = [
            "ETT - Abnormal",
            "ETT - Borderline",
            "ETT - Normal",
            "NGT - Abnormal",
            "NGT - Borderline",
            "NGT - Incompletely Imaged",
            "NGT - Normal",
            "CVC - Abnormal",
            "CVC - Borderline",
            "CVC - Normal",
            "Swan Ganz Catheter Present",
        ]
        self.df.drop(columns=columns_to_remove, inplace=True)


class IndianaDataset(CXRDataset):
    def __init__(self, dataset_path=INDIANA_PATH):
        super().__init__(dataset_path)

    @classmethod
    def get_dataset_name(cls):
        return "Indiana"

    def load_dataset(self):
        logger.debug("Importing Indiana Dataset")
        # Get files from folders
        path = os.path.join(self.dataset_path, "NLMCXR_png")
        files = [file for file in os.listdir(path) if file.endswith(".png")]
        self.df = pd.DataFrame({"filename": files})
        self.df["filepath"] = self.df["filename"].apply(
            lambda x: os.path.join(path, x)
        )

    def preprocess_dataset(self):
        # Load cxr projection from paper csv
        frontal_images = pd.read_csv(
            os.path.join(self.dataset_path, "indiana_frontal_final.csv"),
            header=None,
        )[0].tolist()
        lateral_images = pd.read_csv(
            os.path.join(self.dataset_path, "indiana_lateral_final.csv"),
            header=None,
        )[0].tolist()
        frontal_images = ["CXR" + file + ".png" for file in frontal_images]
        lateral_images = ["CXR" + file + ".png" for file in lateral_images]

        self.df["cxr_projection"] = "PA"
        self.df.loc[
            self.df["filename"].isin(frontal_images), "cxr_projection"
        ] = "PA"
        self.df.loc[
            self.df["filename"].isin(lateral_images), "cxr_projection"
        ] = "L"

        # Extract info from filename. Eg: CXR100_IM-0002-1001.png -> is CXRpatientid_IM-studyid-image_id.png
        # It can also be CXR1_1_IM-0001-3001.png
        self.df["patient_id"] = (
            self.df["filename"].str.extract(r"CXR(\d+)_").astype(int)
        )
        self.df["study_id"] = (
            self.df["filename"].str.extract(r"IM-(\d+)").astype(int)
        )
        self.df["image_id"] = (
            self.df["filename"].str.extract(r"-(\d+).png").astype(int)
        )
        self.df["image_id"] = self.df.apply(
            lambda row: int(
                str(row["patient_id"])
                + str(row["study_id"])
                + str(row["image_id"])
            ),
            axis=1,
        )

        # Extract info from XML files
        self.xml_info = self.parse_all_xml()
        self.df = self.attach_info_from_xml_map(self.df, self.xml_info)

        # Extract all possible labelss:
        self.major_labels = list(
            set(chain.from_iterable(self.df["labels_major"]))
        )
        self.automatic_labels = list(
            set(chain.from_iterable(self.df["labels_automatic"]))
        )

        # Extract pneumothorax from labels:
        self.df["pneumothorax"] = 0
        automatic_pneumothorax = filter_list_by_regex(
            self.automatic_labels, "pneumothorax"
        )
        major_pneumothorax = filter_list_by_regex(
            self.major_labels, "pneumothorax"
        )
        self.df["pneumothorax"] = self.df.apply(
            lambda row: 1
            if any(
                label in automatic_pneumothorax
                for label in row["labels_automatic"]
            )
            or any(
                label in major_pneumothorax for label in row["labels_major"]
            )
            else 0,
            axis=1,
        )

        # Extract normal from labels:
        normal_labels = ["normal"]
        self.df["no_finding"] = self.df.apply(
            lambda row: 1
            if any(label in normal_labels for label in row["labels_major"])
            else 0,
            axis=1,
        )

        # Add dataset name
        self.df["dataset"] = self.get_dataset_name()

        # Common preprocessing steps
        self.preprocessing_common()

        # Remove other columns:
        # self.df.drop(columns=['labels_major', 'labels_automatic'], inplace=True)

    def parse_all_xml(self):
        imageid_to_info = {}

        xml_folder = os.path.join(
            self.dataset_path, "NLMCXR_reports", "ecgen-radiology"
        )

        logger.debug(f"Parsing all XML files in {xml_folder}")

        for fname in tqdm(os.listdir(xml_folder), desc="Parsing XML files"):
            if not fname.endswith(".xml"):
                continue

            xml_path = os.path.join(xml_folder, fname)

            try:
                with open(xml_path, "r", encoding="utf-8") as f:
                    xml_dict = xmltodict.parse(f.read())
                    root = xml_dict["eCitation"]

                    uid = root.get("uId", {}).get("@id", "")

                    # MeSH
                    mesh = root.get("MeSH", {})
                    labels_major = []
                    labels_auto = []

                    if "major" in mesh:
                        major = mesh["major"]
                        if isinstance(major, list):
                            labels_major = [m.lower() for m in major]
                        else:
                            labels_major = [major.lower()]

                    if "automatic" in mesh:
                        auto = mesh["automatic"]
                        if isinstance(auto, list):
                            labels_auto = [a.lower() for a in auto]
                        else:
                            labels_auto = [auto.lower()]

                    # parentImage IDs
                    parent_images = root.get("parentImage", [])
                    if isinstance(parent_images, dict):  # Only one image
                        parent_images = [parent_images]

                    for img in parent_images:
                        img_id = img.get("@id", "")
                        imageid_to_info[img_id] = {
                            "labels_major": sorted(set(labels_major)),
                            "labels_automatic": sorted(set(labels_auto)),
                        }

            except Exception as e:
                print(f"Failed to parse {fname}: {e}")

        return imageid_to_info

    def attach_info_from_xml_map(self, df, xml_data_map):
        df["labels_major"] = ""
        df["labels_automatic"] = ""

        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Attaching info from XML map into dataframe",
        ):
            fname_stem = os.path.splitext(row["filename"])[0]
            if fname_stem in xml_data_map:
                info = xml_data_map[fname_stem]
                df.at[idx, "labels_major"] = info["labels_major"]
                df.at[idx, "labels_automatic"] = info["labels_automatic"]
            else:
                print(f"Warning: No XML entry found for {fname_stem}")

        return df
