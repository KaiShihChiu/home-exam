import cv2
import numpy as np

def find_contours(binary_image):
    """
    Find contours in the binary image.
    
    Args:
        binary_image (np.array): Edge-detected binary image.
    
    Returns:
        list: Contours found in the image.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_valid_contours(contours, min_area=10000):
    """
    Filter contours based on a minimum area threshold.
    
    Args:
        contours (list): List of contours.
        min_area (int): Minimum area to consider a contour as valid.
    
    Returns:
        list: Filtered valid contours.
    """
    valid_contours = []
    areas = []
    perimeters = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            perimeter = cv2.arcLength(contour, True)
            valid_contours.append(contour)
            areas.append(area)
            perimeters.append(perimeter)

    return valid_contours, np.array(areas), np.array(perimeters)
