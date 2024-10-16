import cv2
import numpy as np

class FeatureDetector:
    def detectAndDescribe(self, img):
        """
        Detect and describe features using SIFT.
        
        Args:
            img: Input image (can be grayscale or BGR).
        
        Returns:
            kps: Keypoints detected by SIFT.
            features: Descriptors extracted by SIFT.
        """
        
        if img.dtype != 'uint8':
            img = img.astype(np.uint8)
            
        # check the image shape
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # SIFT detector and descriptor
        sift = cv2.SIFT_create()
        kps, features = sift.detectAndCompute(img, None)
        
        return kps, features

