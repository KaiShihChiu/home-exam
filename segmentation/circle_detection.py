import cv2
import numpy as np


def validate_circles(segmentation_map, circles):
    """
    Ensure that the center of each detected circle lies within the segmentation map.

    Args:
        segmentation_map (np.array): The segmentation map (binary or labeled).
        circles (np.array): Array of detected circles [(x, y, r), ...].

    Returns:
        np.array: Filtered circles with valid center points.
    """
    valid_circles = []

    for (x, y, r) in circles:
        # Convert x and y to integers
        x, y = int(round(x)), int(round(y))

        # Ensure the coordinates are within the bounds of the segmentation map
        if 0 <= y < segmentation_map.shape[0] and 0 <= x < segmentation_map.shape[1]:
            # Check if the circle's center is in a non-zero region of the segmentation map
            if segmentation_map[y, x] > 0:  # Non-zero means valid
                valid_circles.append((x, y, r))

    return np.array(valid_circles)

def detect_circles(segmented_image, dp=1.1, minDist=50, param1=30, param2=25, minRadius=45, maxRadius=90):
    """
    Detect circles using Hough Circle Transform.

    Args:
        segmented_image (np.array): Input 8-bit semantic segmentation image.
        dp (float): Inverse ratio of the accumulator resolution to the image resolution.
        minDist (int): Minimum distance between the centers of detected circles.
        param1 (int): First method-specific parameter for edge detection.
        param2 (int): Second method-specific parameter for circle detection.
        minRadius (int): Minimum circle radius.
        maxRadius (int): Maximum circle radius.

    Returns:
        np.array: Array of detected circles (x, y, r).
    """
    circles = cv2.HoughCircles(segmented_image, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Validate the circles
    if circles is not None:
        valid_circles = validate_circles(segmented_image, circles[0, :])
        return np.round(valid_circles).astype("int")
    else:
        return None


def find_overlapping_circles(circles):
    """
    Find overlapping circles based on their centers and radii.

    Args:
        circles (np.array): Array of circles with (x, y, r).

    Returns:
        list: List of overlapping circle pairs (i, j).
    """
    overlap_pairs = []
    if circles is None:
        print("No circles detected.")
        return overlap_pairs

    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            # Calculate Euclidean distance between centers
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # Check if circles overlap
            if dist < (r1 + r2):  # Adjust factor as needed
                overlap_pairs.append((i, j))
                print(f"Circles {i} and {j} are overlapping. Distance: {dist:.2f}")

    return overlap_pairs

def draw_circles(image, circles, overlap_pairs):
    """
    Draw detected circles and highlight overlapping ones.

    Args:
        image (np.array): Original image to draw circles on.
        circles (np.array): Array of circles with (x, y, r).
        overlap_pairs (list): List of overlapping circle pairs (i, j).

    Returns:
        np.array: Image with drawn circles.
    """
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw all circles in green
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)

    # Highlight overlapping circles in red
    for (i, j) in overlap_pairs:
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        cv2.circle(output, (x1, y1), r1, (0, 0, 255), 3)
        cv2.circle(output, (x2, y2), r2, (0, 0, 255), 3)

    return output
