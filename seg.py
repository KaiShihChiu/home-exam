from segmentation.image_processing import read_and_convert_image, apply_noise_reduction, detect_edges, morphological_processing
from segmentation.contour_analysis import find_contours, filter_valid_contours
from segmentation.clustering import perform_kmeans, apply_ransac
from segmentation.segmentation import create_semantic_segmentation, apply_color_map
from segmentation.visualization import display_image, plot_histogram
from segmentation.circle_detection import detect_circles, find_overlapping_circles, draw_circles

import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Image processing
    gray_image = read_and_convert_image('img/panorama_2.jpg')
    blurred_image = apply_noise_reduction(gray_image)
    edges = detect_edges(blurred_image)
    processed_image = morphological_processing(edges)

    # 2. Contour analysis
    contours = find_contours(processed_image)
    valid_contours, areas, perimeters = filter_valid_contours(contours)

    # 3. Clustering and RANSAC
    data = np.array(list(zip(areas, perimeters)))
    best_k, labels, _ = perform_kmeans(data)
    inlier_mask, ransac_model, inlier_data = apply_ransac(data)

    # 4. Semantic segmentation
    segmentation_image = create_semantic_segmentation(valid_contours, labels, gray_image.shape)
    colored_segmentation = apply_color_map(segmentation_image, best_k)

    # 5. Visualization
    # display_image(colored_segmentation, "Semantic Segmentation")
    # plot_histogram(areas, "Contour Areas", "Area", "Count")
    
    # Load the segmented image (8-bit semantic segmentation result)
    semantic_segmentation_8bit = (segmentation_image * (255 // best_k)).astype(np.uint8)

    # Detect circles
    circles = detect_circles(semantic_segmentation_8bit, dp=1.2, minDist=45, param1=50, param2=10, minRadius=50, maxRadius=80)

    # Find overlapping circles
    overlap_pairs = find_overlapping_circles(circles)

    # Draw circles and highlight overlaps
    color_seg = draw_circles(semantic_segmentation_8bit, circles, overlap_pairs)

    # Display the result
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(semantic_segmentation_8bit, cmap='gray'), plt.title("Original Segmentation")
    plt.subplot(122), plt.imshow(color_seg), plt.title("Detected Circles with Overlaps")
    plt.show()
    
    
if __name__ == "__main__":
    main()
