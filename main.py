import sys
import os

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from descriptors.dominant_color import DominantColorDescriptor
from descriptors.color_layout import ColorLayoutDescriptor
from descriptors.color_structure import ColorStructureDescriptor

DESCRIPTOR_LIST = {
    "DCD": DominantColorDescriptor(),
    "CLD": ColorLayoutDescriptor(),
    "CSD": ColorStructureDescriptor()
}


def is_image(file_name):
    """
    Checks if a file is an image based on its extension.

    Parameters:
        file_name (str): Name of the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def get_images_subdirectory(directory_path):
    """
    Retrieves a list of image files within a directory and its subdirectories.

    Parameters:
        directory_path (str): Path to the directory.

    Returns:
        list: List of paths to image files.
    """
    return [os.path.join(root, file) for root, _, files in os.walk(directory_path)
            for file in files if is_image(file)]


def train(train_directory, descriptor_used, output_file):
    """
    Uses selected descriptor on a set of images and saves the results to a file.

    Parameters:
        train_directory (str): Path to the directory containing training images.
        descriptor_used (str): Key specifying the descriptor to be used.
        output_file (str): Path to the output file.
    """
    start_time = time.time()
    train_image_list = get_images_subdirectory(train_directory)
    n_images = len(train_image_list)

    with open(output_file, 'w') as f:
        for idx, image_file in enumerate(train_image_list):
            print(f"{100 * idx / n_images:.1f}/100% completed\n"
                  f"{time.time() - start_time:.1f}s elapsed\n", end='')
            img = cv2.imread(image_file)
            descriptor = DESCRIPTOR_LIST[descriptor_used].get_descriptor(img)
            f.write(f"{image_file} {descriptor}\n")

            # Clear the line for progression
            print('\033[1A', end='\x1b[2K')
            print('\033[1A', end='\x1b[2K')

    end_time = time.time() - start_time
    print(f"\nUsed '{descriptor_used}' descriptors on files from the folder '{train_directory}'. "
          f"Results are saved in '{output_file}'.")
    print(f"{n_images} images processed in {end_time:.1f} s, {end_time / n_images:.3f} per file")


def iterate_file_lines(database_file):
    """
    Iterates over lines in a database file containing descriptors.

    Parameters:
        database_file (str): Path to the database file.

    Yields:
        tuple: File name and descriptor values.
    """
    try:
        with open(database_file, 'r') as f:
            for descriptor in f:
                descriptor_line = descriptor.strip()
                descriptor_file_name, descriptor_values = descriptor_line.split(" ", 1)
                yield descriptor_file_name, descriptor_values
    except FileNotFoundError:
        print(f"File '{database_file}' is unknown.")


def get_prediction(query_img, descriptor_used, descriptor_database):
    """
    Retrieves predictions for a query image based on descriptors in a database.

    Parameters:
        query_img (numpy.ndarray): Query image array.
        descriptor_used (str): Key specifying the descriptor to be used.
        descriptor_database (str): Path to the descriptor database file.

    Returns:
        list: Predicted labels for the query image.
    """
    distances_list = []
    descriptor_query_img = DESCRIPTOR_LIST[descriptor_used].get_descriptor(query_img)

    for file_name, descriptor in iterate_file_lines(descriptor_database):
        d_1 = np.array(descriptor_query_img.split(), dtype=int)
        d_2 = np.array(descriptor.split(), dtype=int)
        distance = DESCRIPTOR_LIST[descriptor_used].get_distance(d_1, d_2)
        distances_list.append((file_name, distance))

    # Keep the 5 smaller distances
    distances_list.sort(key=lambda x: x[1])
    distances_list = distances_list[:5]

    return distances_list


def evaluate(test_directory, descriptor_used, descriptor_database):
    """
    Evaluates the performance of a descriptor on a test dataset.

    Parameters:
        test_directory (str): Path to the directory containing test images.
        descriptor_used (str): Key specifying the descriptor to be used.
        descriptor_database (str): Path to the descriptor database file.
    """
    y_true = []
    y_pred = []

    test_image_list = get_images_subdirectory(test_directory)
    n_images = len(test_image_list)

    start_time = time.time()

    for idx, image_file in enumerate(test_image_list):
        print(f"{100 * idx / n_images:.1f}/100% completed\n"
              f"{time.time() - start_time:.1f}s elapsed\n", end='')
        true_label = image_file.split("/")[-2]
        query_img = cv2.imread(image_file)

        y_true.append(true_label)

        distances_list = get_prediction(query_img, descriptor_used, descriptor_database)
        pred_labels = [distances[0].split("/")[-2] for distances in distances_list]

        y_pred.append(pred_labels)

        # Clear the line for progression
        print('\033[1A', end='\x1b[2K')
        print('\033[1A', end='\x1b[2K')

    end_time = time.time() - start_time

    print(f"{n_images} images processed in {end_time:.1f} s, {end_time / n_images:.3f} per file")
    print(f"The accuracy of the {descriptor_used} on the {test_directory} folder "
          f"is {calculate_mean_average_precision(y_pred, y_true) * 100:.2f}%")


def calculate_mean_average_precision(batch_pred, batch_true):
    """
    Calculates the mean average precision (mAP) for a set of predictions.

    Parameters:
        batch_pred (list): Predicted labels for each query image.
        batch_true (list): True labels for each query image.

    Returns:
        float: Mean average precision.
    """
    total_map = 0
    num_batches = len(batch_true)
    for batch_idx in range(num_batches):
        y_true = batch_true[batch_idx]
        y_pred = batch_pred[batch_idx]
        num_correct = 0
        total_precision = 0
        for i in range(5):
            if y_pred[i] == y_true:
                num_correct += 1
                precision = num_correct / (i + 1)
                total_precision += precision
        if num_correct == 0:
            avg_precision = 0
        else:
            avg_precision = total_precision / num_correct

        total_map += avg_precision
    return total_map / num_batches


def display_match(img_name, descriptor_used, descriptor_database):
    """
    Displays matching images for a query image.

    Parameters:
        img_name (str): Path to the query image.
        descriptor_used (str): Key specifying the descriptor to be used.
        descriptor_database (str): Path to the descriptor database file.
    """
    query_img = cv2.imread(img_name)

    fig = plt.figure(figsize=(15, 7))
    add_image_to_plot(fig, 1, query_img, "Query image")

    distances_list = get_prediction(query_img, descriptor_used, descriptor_database)
    pred_labels = [distances[0].split("/")[-2] for distances in distances_list]

    for i in range(5):
        match_image = cv2.imread(distances_list[i][0])

        add_image_to_plot(fig, i + 2, match_image, f"Match {i + 1}", pred_labels[i])

    plt.show()
    plt.close(fig)


def add_image_to_plot(fig, n, img, title, label=None):
    """
    Adds an image to a plot.

    Parameters:
        fig (matplotlib.figure.Figure): Figure object.
        n (int): Position index for the subplot.
        img (numpy.ndarray): Image array.
        title (str): Title for the subplot.
        label (str, optional): Label for the image. Defaults to None.
    """
    ax = fig.add_subplot(2, 3, n)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    if label:
        plt.text(0.5, -0.03, label,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, color='black', fontsize=8)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py [train/eval/query] [directory] [descriptor] [output_file]")
        sys.exit(1)

    mode = sys.argv[1]
    path_name = sys.argv[2]
    descriptor_method = sys.argv[3]
    descriptor_list = sys.argv[4]

    if not os.path.exists(path_name):
        print(f"The directory '{path_name}' does not exist.")
        sys.exit(1)

    if descriptor_method not in list(DESCRIPTOR_LIST.keys()):
        print(f"The descriptor '{descriptor_method}' is unknown.")
        sys.exit(1)

    if mode == "train":
        train(path_name, descriptor_method, descriptor_list)
    elif mode == "eval":
        evaluate(path_name, descriptor_method, descriptor_list)
    elif mode == "query":
        display_match(path_name, descriptor_method, descriptor_list)
    else:
        print("The mode must be 'train', 'eval' or 'query'.")
        sys.exit(1)
