import cv2
import matplotlib.pyplot as plt
from stitching.stitcher import stitch_multiple_images
from stitching.utils import load_images

def main():
    # Define the list of image filenames for stitching (no pairs, just a list of images in order)
    src_path = "img"
    num_img = 3

    # # Load the images
    images = load_images(src_path, num_img, resize=True, scale_percent=50)
    
    if not images:
        print("No valid images to stitch.")
        return
    
    # show the dimensions of the images
    for i, img in enumerate(images):
        print(f"Image {i + 1}: {img.shape[1]} x {img.shape[0]}")
    
    # Perform stitching across all images
    panorama = stitch_multiple_images(images, blending_mode="linearBlendingWithConstant")
    
    # # Display the final stitched image
    plt.figure()
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title("Final Panorama Image")
    plt.show()  # Ensure the plot window stays open until manually closed

if __name__ == "__main__":
    main()
