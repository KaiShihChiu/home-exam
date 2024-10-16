import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def resizing(img, scale_percent=20):
    """
    Resize the given image based on the provided scaling percentage.

    Args:
        img (np.array): Input image.
        scale_percent (int, optional): Percentage to scale the image by. Default is 20%.

    Returns:
        np.array: Resized image.
    """
    if img is None:
        raise ValueError("Input image is invalid.")

    # Calculate new dimensions based on the scaling percentage
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image using INTER_AREA interpolation for shrinking
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized_img


def load_images(src_path="img", num_img=5, resize=False, scale_percent=20):
    """
    Load multiple images from a specified folder, with an option to resize them.

    Args:
        src_path (str): Path to the folder containing images.
        num_img (int, optional): Number of images to load. Default is 5.
        resize (bool, optional): Whether to resize the images. Default is False.
        scale_percent (int, optional): Percentage to scale the image by (if resize is True). Default is 20%.

    Returns:
        list: A list of loaded (and optionally resized) images.
    """
    # Check if the source path exists
    if not os.path.exists(src_path):
        raise ValueError(f"The specified path does not exist: {src_path}")
    
    # Retrieve all image filenames in the folder with the specified extensions
    image_files = [f for f in os.listdir(src_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(image_files)
    
    # Sort the filenames based on the numeric part of the filename (e.g., IMG_7111)
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    image_files = image_files[:num_img]
    

    if len(image_files) == 0:
        print(f"No image files found in the folder: {src_path}")
        return

    images = []
    for file in image_files:
        # Read the image
        img = cv2.imread(os.path.join(src_path, file), cv2.IMREAD_COLOR)

        if img is None:
            print(f"Error loading image: {file}")
            continue

        # Resize the image if the resize flag is True
        if resize:
            img = resizing(img, scale_percent)

        # Convert BGR to RGB for proper display using matplotlib
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    # Display the images using matplotlib
    fig, axes = plt.subplots(1, len(images), figsize=(20, 10))
    plt.title("Original Images")

    for ax, img, img_name in zip(axes, images, image_files):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_name}")  # Set the subtitle for each image
        ax.axis('off')
    # plt.show()
    return images
    
def largest_rectangle_area(heights):
    """
    Given a list of histogram heights, compute the largest rectangle area.
    This is a helper function that uses a stack to calculate the largest rectangle.
    
    Args:
        heights (list): Heights of the histogram.
    
    Returns:
        int: The largest rectangle area.
        (int, int, int, int): Coordinates of the rectangle (height, left, right, bottom).
    """
    stack = []
    max_area = 0
    left_index = 0
    max_rect = (0, 0, 0, 0)  # height, left, right, bottom
    
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                left_index = stack[-1] + 1 if stack else 0
                max_rect = (height, left_index, i - 1, height)
        stack.append(i)
    
    return max_area, max_rect

def find_largest_rectangle(thresh):
    """
    Find the largest rectangle inside the white area of the binary mask.
    
    Args:
        thresh (np.array): Binary thresholded image (with 0 and 255 values).
    
    Returns:
        tuple: Coordinates (top, bottom, left, right) of the largest rectangle.
    """
    # Initialize variables
    rows, cols = thresh.shape
    heights = [0] * cols
    max_area = 0
    max_rectangle = (0, 0, 0, 0)  # top, bottom, left, right

    # Iterate through each row and calculate histogram heights
    for row in range(rows):
        for col in range(cols):
            # If the pixel is white (255), increment the height, otherwise reset it to 0
            if thresh[row, col] == 255:
                heights[col] += 1
            else:
                heights[col] = 0
        
        # Find the largest rectangle for the current row's histogram
        area, (height, left, right, bottom) = largest_rectangle_area(heights)
        
        if area > max_area:
            max_area = area
            max_rectangle = (row - height + 1, row, left, right)
    
    return max_rectangle

def remove_black_borders(img):
    """
    Remove black borders from the image and find the largest rectangle inside the white area.
    
    Args:
        img (np.array): Input color image (BGR).
    
    Returns:
        np.array: Cropped image with the largest rectangle area.
    """
    # Ensure the image is unsigned 8-bit
    if img.dtype != 'uint8':
        img = img.astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where non-black pixels are white (255) and black pixels are black (0)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up small noise (optional)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the largest rectangle in the white region
    top, bottom, left, right = find_largest_rectangle(thresh)
    
    # Crop the image based on these boundaries
    cropped_img = img[top:bottom+1, left:right+1]
    
    return cropped_img