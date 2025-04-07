"""
This module contains the function to process the contour in the image.
"""
from typing import Tuple
import cv2
import numpy as np


def process_contour(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the contour in the image using adaptive thresholding and robust contour detection.
    Returns the processed image and the largest valid contour found.
    """
    try:
        # Ensure image is in correct format
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding for better text/background separation
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        # Find contours with different parameters for better accuracy
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # If no contours found, return original image and empty contour
            return image, np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]])

        # Filter out noise and get valid contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        if not valid_contours:
            return image, np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]])

        # Get the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Create mask and apply morphological operations
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply the mask to the image
        if len(image.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image[mask > 0] = 255

        return image, largest_contour

    except Exception as e:
        print(f"Error in process_contour: {str(e)}")
        return image, np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]])
