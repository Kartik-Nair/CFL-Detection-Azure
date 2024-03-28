import cv2
import numpy as np
from PIL import Image


def filter_contours_for_straight_lines(contours, min_length=100, threshold=50):
    filtered_contours = []
    for contour in contours:
        # Consider using arcLength to filter based on contour length without assuming straightness
        if cv2.arcLength(contour, True) > min_length:
            filtered_contours.append(contour)
    return filtered_contours


def process_contour_image(contour_image):
    # Convert contour image to numpy array
    image_data = np.array(contour_image)

    # Apply threshold to get binary image
    thresholded = (image_data > 0) * 255

    # Find the first and last white pixel in each column
    first_white_pixels = np.argmax(thresholded, axis=0)
    last_white_pixels = thresholded.shape[0] - np.argmax(thresholded[::-1], axis=0) - 1

    # Create a new array to hold the processed image
    processed_image_data = np.zeros_like(image_data)

    # Mark only the first and last white pixel in the column as white
    for col in range(thresholded.shape[1]):
        if (
            first_white_pixels[col] != 0
            or last_white_pixels[col] != thresholded.shape[0] - 1
        ):
            processed_image_data[first_white_pixels[col], col] = 255
            if (
                first_white_pixels[col] != last_white_pixels[col]
            ):  # If they are not the same pixel
                processed_image_data[last_white_pixels[col], col] = 255

    # Convert processed data back to an image
    processed_contour_image = Image.fromarray(processed_image_data.astype(np.uint8))
    return processed_contour_image, first_white_pixels, last_white_pixels


def detect_wall_edge(file):
    image = Image.open(file).convert("RGB").resize((256, 256))
    image = np.array(image)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the median pixel value of the grayscale image
    v = np.median(gray_image)

    # Set lower and upper thresholds for Canny dynamically
    sigma = 0.5
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Apply Gaussian blur
    # Adaptive kernel size based on image size (smaller images might need a smaller kernel)
    kernel_size = (3, 3) if image.shape[0] < 500 else (5, 5)

    # Apply Gaussian blur with adaptive kernel size
    blurred = cv2.GaussianBlur(gray_image, kernel_size, 0)

    # Apply Canny edge detection using Otsu's thresholds
    edges = cv2.Canny(blurred, lower, upper)

    # Use dilation to close gaps in the edges
    # Adjust dilation parameters
    kernel = np.ones((1, 1), np.uint8)  # Smaller kernel
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the dilated Canny edges
    contours, _ = cv2.findContours(
        dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter for straight lines
    filtered_contours = filter_contours_for_straight_lines(contours)

    # Create an empty mask to draw the contours on
    vessel_wall_contours = np.zeros_like(gray_image)

    # Draw the filtered contours - this will be the vessel walls
    cv2.drawContours(vessel_wall_contours, filtered_contours, -1, (255), thickness=1)
    # vessel_wall_contours_resized = cv2.resize(vessel_wall_contours, (256, 256))

    # Convert the contours array to an image
    vessel_wall_contours_image = Image.fromarray(vessel_wall_contours)

    # Process the contour image with the provided function
    processed_vessel_wall_contours_image, first_white_pixels, last_white_pixels = (
        process_contour_image(vessel_wall_contours_image)
    )

    return (
        processed_vessel_wall_contours_image,
        first_white_pixels,
        last_white_pixels,
    )
