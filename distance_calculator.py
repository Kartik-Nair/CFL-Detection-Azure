import cv2
import numpy as np


def find_closest_non_outlier(arr, index, outlier_indices):
    left_index = index - 1
    right_index = index + 1

    while left_index >= 0 or right_index < len(arr):
        if left_index >= 0 and not outlier_indices[left_index]:
            return arr[left_index]
        if right_index < len(arr) and not outlier_indices[right_index]:
            return arr[right_index]
        left_index -= 1
        right_index += 1

    # If no non-outlier value is found, return the mean of the array
    return np.mean(arr)


def replace_outliers(arr):
    # Calculate the mean and standard deviation of the array
    mean = np.mean(arr)
    std_dev = np.std(arr)

    # Define the threshold for identifying outliers
    threshold = 2.0  # You can adjust this value based on your data distribution

    # Find the indices of outliers
    outlier_indices = np.abs(arr - mean) > threshold * std_dev

    # Replace outliers with the closest non-outlier value
    for i in range(len(arr)):
        if outlier_indices[i]:
            arr[i] = find_closest_non_outlier(arr, i, outlier_indices)

    return arr


def find_core(image):
    # Set lower and upper thresholds for Canny dynamically
    v = np.median(image)
    sigma = 0.5
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # Apply Canny edge detection using Otsu's thresholds

    edges = cv2.Canny(np.uint8((image > 0.9)), lower, upper)

    first_white_pixels = np.argmax(edges, axis=0)
    fixed_first_white_pixels = replace_outliers(first_white_pixels)

    last_white_pixels = edges.shape[0] - np.argmax(edges[::-1], axis=0) - 1
    fixed_last_white_pixels = replace_outliers(last_white_pixels)

    return fixed_first_white_pixels, fixed_last_white_pixels


def calculate_distance(core_img, first_white_pixels_wall, last_white_pixels_wall):
    first_white_pixels_core, last_white_pixels_core = find_core(core_img)
    distance_top = np.abs(first_white_pixels_wall - first_white_pixels_core)
    distance_bottom = np.abs(last_white_pixels_core - last_white_pixels_wall)
    return distance_top + distance_bottom
