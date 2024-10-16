from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .homography import Homography
from .blending import Blender
import numpy as np
import cv2

class Stitcher:
    def __init__(self):
        pass
    
    def stitch(self, imgs, blending_mode="linearBlending", ratio=0.8):
        """
        Stitch two images together using keypoint detection, homography estimation, and blending.
        
        Args:
            imgs (list): List containing two images (left and right) to stitch.
            blending_mode (str, optional): Blending mode for the final image, default is "linearBlending".
            ratio (float, optional): Ratio for Lowe's ratio test in keypoint matching. Default is 0.75.
        
        Returns:
            np.array: The final stitched image.
        """
        # Extract left and right images
        img_left, img_right = imgs
        print(f"Left img size: {img_left.shape[:2]}, Right img size: {img_right.shape[:2]}")
        
        # Step 1: Detect keypoints and extract features using SIFT
        print("Step 1 - Detect keypoints and extract features with SIFT detector...")
        detector = FeatureDetector()  # Assume you have a FeatureDetector class for this
        kps_l, features_l = detector.detectAndDescribe(img_left)
        kps_r, features_r = detector.detectAndDescribe(img_right)

        # Step 2: Match keypoints between the two images
        print("Step 2 - Matching keypoints between left and right images...")
        matcher = FeatureMatcher()  # Assume you have a FeatureMatcher class for this
        matches_pos = matcher.matchKeyPoint(kps_l, kps_r, features_l, features_r, ratio)
        print(f"Number of matching points: {len(matches_pos)}")
        matcher.drawMatches([img_left, img_right], matches_pos)  # Visualize matching points
        
        # Step 3: Estimate homography using RANSAC
        print("Step 3 - Estimating homography matrix using RANSAC algorithm...")
        homography = Homography()  # Assume Homography class is implemented
        HomoMat = homography.fitHomoMat(matches_pos)
        
        # Step 4: Warp the right image based on the homography matrix and blend
        print("Step 4 - Warping and stitching images together...")
        warp_img = self.warp([img_left, img_right], HomoMat, blending_mode)
        
        return warp_img
    
    def warp(self, imgs, HomoMat, blending_mode):
        """
        Warp the second image to align with the first image based on the homography matrix.
        
        Args:
            imgs (list): List of two images (left and right).
            HomoMat (np.array): Homography matrix for warping.
            blending_mode (str): Blending mode to use for stitching.
        
        Returns:
            np.array: The final stitched image.
        """
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        
        # Create an empty image that can hold the stitched result
        stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="int")
        inv_H = np.linalg.inv(HomoMat)  # Inverse homography for warping right image
        
        # Apply homography to warp the right image onto the left image
        for i in range(stitch_img.shape[0]):
            for j in range(stitch_img.shape[1]):
                coor = np.array([j, i, 1])
                img_right_coor = inv_H @ coor
                img_right_coor /= img_right_coor[2]
                y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1]))
                if 0 <= x < hr and 0 <= y < wr:
                    stitch_img[i, j] = img_right[x, y]
        
        # Blending the two images together
        blender = Blender()  # Assume you have a Blender class for this
        if blending_mode == "linearBlending":
            stitch_img = blender.linearBlending([img_left, stitch_img])
        elif blending_mode == "linearBlendingWithConstant":
            stitch_img = blender.linearBlendingWithConstantWidth([img_left, stitch_img])

        return stitch_img

def stitch_multiple_images(images, blending_mode="linearBlendingWithConstant"):
    """
    Stitch multiple images together by sequentially stitching pairs of images.
    
    Args:
        images (list): List of images to be stitched.
        blending_mode (str): The blending mode to use for stitching.
    
    Returns:
        np.array: The final stitched panoramic image.
    """
    stitcher = Stitcher()
    stitched_img = images[0]  # Start with the first image as the base
    
    # Iterate through the rest of the images and stitch each to the current stitched result
    for i in range(1, len(images)):
        print(f"Stitching image {i + 1}...")

        # Perform stitching on the resized images
        stitched_img = stitcher.stitch([stitched_img, images[i]], blending_mode)

        # Removing black borders after stitching
        from stitching.utils import remove_black_borders  # Remove borders after stitching
        stitched_img = remove_black_borders(stitched_img)
        
        # save stitched image
        saveFilePath = f"img/panorama_{i}.jpg"
        cv2.imwrite(saveFilePath, stitched_img)
        
        
    return stitched_img
