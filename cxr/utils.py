import copy
import logging
from PIL import Image
from matplotlib import pyplot as plt
import re


def create_opacity_label(
    all_labels: dict, opacity_labels: list, opacity_name: str = "opacity"
) -> dict:
    """Creates opacity label by modifying the original label"""
    all_labels = copy.deepcopy(all_labels)
    for orig_key, orig_label in all_labels.items():
        if orig_label in opacity_labels:
            all_labels[orig_key] = opacity_name
    return all_labels


def show_sample_images(
    df, path_column, nrows: int = 3, ncols: int = 3, title: str = ""
):
    try:
        selected_files = df[path_column].sample(n=nrows * ncols).tolist()
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3)
        )
        fig.suptitle(title, fontsize=20)
        axes = axes.flatten()

        for idx, file in enumerate(selected_files):
            img = Image.open(file)
            axes[idx].imshow(img, cmap="gray")
            axes[idx].axis("off")
            axes[idx].set_title(f"Image {idx + 1}")

        # Turn off any unused subplots
        for ax in axes[len(selected_files) :]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    except ValueError:
        logging.error(
            f"Not enough samples to show ({df.shape[0]} samples < {ncols * nrows} needed)"
        )
    except Exception as e:
        logging.error(f"Error plotting images: {e}")


def search_pattern(strings, pattern):
    """
    Search for a pattern in a list of strings and return matches.

    Args:
        strings (list of str): List of strings to search in.
        pattern (str): Regular expression pattern to search for.

    Returns:
        list of str: Strings that match the pattern.
    """
    matches = []
    compiled_pattern = re.compile(
        pattern
    )  # Compile the regex pattern for efficiency
    for string in strings:
        if compiled_pattern.search(
            string
        ):  # Check if the pattern matches the string
            matches.append(string)
    return matches


def check_images(list_of_files):
    corrupted_images = []
    logging.info(f"Checking {len(list_of_files)} images for corruption")

    for filename in list_of_files:
        try:
            with Image.open(filename) as img:
                img.verify()  # Check for corruption without loading full image
        except (IOError, SyntaxError, Image.UnidentifiedImageError):
            logging.info(f"Corrupted image detected: {filename}")
            corrupted_images.append(filename)

    logging.info(f" Num corrupted images: {len(corrupted_images)}")

    return corrupted_images


def show_generator(
    generator, num_batches=1, mode="binary", cmap="gray", figsize=(12, 12)
):
    for n in range(num_batches):
        x_aug = next(generator)
        print("Shape: ", x_aug.shape)

        if mode == "binary":
            for i in range(x_aug.shape[0]):
                plt.imshow(x_aug[i], cmap=cmap)
                plt.show()

        if mode == "grayscale":
            for i in range(x_aug.shape[0]):
                plt.imshow(x_aug[i], cmap=cmap)
                plt.show()

        if mode == "rgb":
            for i in range(x_aug.shape[0]):
                plt.imshow(x_aug[i])
                plt.show()

        if mode == "multichannel":
            for i in range(x_aug.shape[0]):
                fig, ax = plt.subplots(ncols=x_aug.shape[3], figsize=figsize)
                for ch in range(x_aug.shape[3]):
                    ax[ch].imshow(x_aug[i, :, :, ch], cmap=cmap)
                plt.show()


def print_dictonary_keys(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            new_path = f"{path}/{k}"
            print(new_path)
            print_dictonary_keys(v, new_path)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            print_dictonary_keys(item, f"{path}[{i}]")


def search_dictionary_key(d, target_key, path=""):
    results = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_path = f"{path}/{k}"
            if k == target_key:
                results.append((new_path, v))
            results.extend(search_dictionary_key(v, target_key, new_path))
    elif isinstance(d, list):
        for i, item in enumerate(d):
            results.extend(
                search_dictionary_key(item, target_key, f"{path}[{i}]")
            )
    return results


def search_dictionary_key_by_regex(d, pattern, path=""):
    results = []
    regex = re.compile(pattern)

    if isinstance(d, dict):
        for k, v in d.items():
            new_path = f"{path}/{k}"
            if regex.search(k):
                results.append((new_path, v))
            results.extend(
                search_dictionary_key_by_regex(v, pattern, new_path)
            )

    elif isinstance(d, list):
        for i, item in enumerate(d):
            results.extend(
                search_dictionary_key_by_regex(item, pattern, f"{path}[{i}]")
            )

    return results


def filter_list_by_regex(lst, pattern):
    regex = re.compile(pattern)
    return [item for item in lst if regex.search(item)]


def check_gpu(names=False, computation=False):
    import tensorflow as tf

    print("tf version: ", tf.__version__)
    print("Number of GPUs: ", len(tf.config.list_physical_devices("GPU")))
    print("Device names: ", tf.config.list_physical_devices("GPU"))
    print("CUDA support: ", tf.test.is_built_with_cuda())
    if names:
        from tensorflow.python.client import device_lib

        print(device_lib.list_local_devices())
    if computation:
        tf.debugging.set_log_device_placement(True)
        print(
            "Physical devices: \n",
            tf.config.experimental.list_physical_devices("GPU"),
        )
        print(
            "Logical devices: \n",
            tf.config.experimental.list_logical_devices(),
        )

        # Create some tensors
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(c)
