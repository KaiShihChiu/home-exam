import cv2
import numpy as np

def create_semantic_segmentation(valid_contours, labels, image_shape):
    """
    Create a semantic segmentation map by assigning unique labels to different clusters.
    
    Args:
        valid_contours (list): List of valid contours.
        labels (list): Corresponding cluster labels for each contour.
        image_shape (tuple): Shape of the original image.
    
    Returns:
        np.array: Semantic segmentation image with labeled clusters.
    """
    segmentation_image = np.zeros(image_shape, dtype=np.uint8)
    
    for i, contour in enumerate(valid_contours):
        label = labels[i] + 1  # Ensure each cluster has a unique label, starting from 1
        cv2.drawContours(segmentation_image, [contour], -1, int(label), thickness=cv2.FILLED)

    return segmentation_image

def apply_color_map(segmentation_image, num_clusters):
    """
    Apply a color map to the semantic segmentation image for visualization.
    
    Args:
        segmentation_image (np.array): Semantic segmentation image.
        num_clusters (int): Number of clusters.
    
    Returns:
        np.array: Colored segmentation image.
    """
    max_value = 255 // num_clusters
    segmentation_8bit = (segmentation_image * max_value).astype(np.uint8)
    colored_segmentation = cv2.applyColorMap(segmentation_8bit, cv2.COLORMAP_JET)
    return colored_segmentation
