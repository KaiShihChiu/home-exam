import cv2
import numpy as np

def read_and_convert_image(image_path):
    """
    Reads the image from the given path and converts it to grayscale.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        np.array: Grayscale image.
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def apply_noise_reduction(gray_image):
    """
    Apply noise reduction using bilateral filter.
    
    Args:
        gray_image (np.array): Grayscale image.
    
    Returns:
        np.array: Blurred image.
    """
    blurred_image = cv2.bilateralFilter(gray_image, d=15, sigmaColor=75, sigmaSpace=75)
    return blurred_image

def detect_edges(blurred_image):
    """
    Perform edge detection on the blurred image using Canny edge detection.
    
    Args:
        blurred_image (np.array): Blurred grayscale image.
    
    Returns:
        np.array: Edge-detected binary image.
    """
    edges = cv2.Canny(blurred_image, 20, 50)
    return edges

def morphological_processing(edges):
    """
    Perform morphological operations to remove noise and connect contours.
    
    Args:
        edges (np.array): Edge-detected binary image.
    
    Returns:
        np.array: Processed binary image with dilated and eroded edges.
    """
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=4)
    eroded = cv2.erode(dilated, kernel, iterations=4)
    return eroded
