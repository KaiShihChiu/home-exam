import numpy as np
import matplotlib.pyplot as plt

class Blender:
    def linearBlending(self, imgs):
        '''
        Linear Blending (also known as Feathering)
        This function blends two overlapping images smoothly using an alpha mask.
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]  # Dimensions of the left image
        (hr, wr) = img_right.shape[:2]  # Dimensions of the right image

        # Create masks for the left and right images to identify non-zero regions
        img_left_mask = np.ones((hr, wr), dtype="int")
        img_right_mask = np.ones((hr, wr), dtype="int")
        
        # Identify non-zero pixels in the left image
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1

        # Identify non-zero pixels in the right image
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # Create an overlap mask to mark the overlapping region between both images
        overlap_mask = np.ones((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if img_left_mask[i, j] > 0 and img_right_mask[i, j] > 0:
                    overlap_mask[i, j] = 1
        
        # Display the overlap mask
        plt.figure(21)
        plt.title("Overlap Mask")
        plt.imshow(overlap_mask.astype(int), cmap="gray")
        
        # Create an alpha mask for linear blending
        alpha_mask = np.ones((hr, wr))  # Initialize with all ones

        # Iterate over each row to calculate alpha values for blending
        for i in range(hr): 
            minIdx = maxIdx = -1  # Track the min and max overlap indices in the row
            for j in range(wr):
                if overlap_mask[i, j] == 1 and minIdx == -1:
                    minIdx = j  # First overlap pixel in the row
                if overlap_mask[i, j] == 1:
                    maxIdx = j  # Last overlap pixel in the row
            
            # Skip if there is no valid overlap region
            if minIdx == maxIdx:
                continue

            # Calculate the step size for alpha blending
            decrease_step = 1 / (maxIdx - minIdx)

            # Assign alpha values decreasing linearly across the overlap region
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        # Create a copy of the right image to store the blended result
        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)

        # Apply linear blending to the overlapping region
        for i in range(hr):
            for j in range(wr):
                if overlap_mask[i, j] > 0:
                    linearBlending_img[i, j] = (
                        alpha_mask[i, j] * img_left[i, j] + 
                        (1 - alpha_mask[i, j]) * img_right[i, j]
                    )
        
        return linearBlending_img

    def linearBlendingWithConstantWidth(self, imgs):
        '''
        Linear Blending with Constant Width (to avoid ghost regions)
        This function blends two overlapping images but restricts blending to a constant-width region.
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]  # Dimensions of the left image
        (hr, wr) = img_right.shape[:2]  # Dimensions of the right image

        # Initialize masks for both images
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_width = 3  # Define a constant width for blending

        # Mark non-zero pixels in the left image mask
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1

        # Mark non-zero pixels in the right image mask
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1

        # Create an overlap mask to mark the overlapping region between both images
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if img_left_mask[i, j] > 0 and img_right_mask[i, j] > 0:
                    overlap_mask[i, j] = 1

        # Initialize the alpha mask with zeros
        alpha_mask = np.zeros((hr, wr))

        # Iterate over each row to assign alpha values in the overlap region
        for i in range(hr):
            minIdx = maxIdx = -1  # Track the min and max overlap indices in the row
            for j in range(wr):
                if overlap_mask[i, j] == 1 and minIdx == -1:
                    minIdx = j  # First overlap pixel in the row
                if overlap_mask[i, j] == 1:
                    maxIdx = j  # Last overlap pixel in the row

            # Skip if there is no valid overlap region
            if minIdx == maxIdx:
                continue

            # Calculate the step size for alpha blending
            decrease_step = 1 / (maxIdx - minIdx)

            # Find the middle index of the overlap region
            middleIdx = int((maxIdx + minIdx) / 2)

            # Assign alpha values to the left side of the middle index
            for j in range(minIdx, middleIdx + 1):
                if j >= middleIdx - constant_width:
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 1  # Fully left image

            # Assign alpha values to the right side of the middle index
            for j in range(middleIdx + 1, maxIdx + 1):
                if j <= middleIdx + constant_width:
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 0  # Fully right image

        # Create a copy of the right image to store the blended result
        linearBlendingWithConstantWidth_img = np.copy(img_right)
        linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)

        # Apply linear blending with constant width
        for i in range(hr):
            for j in range(wr):
                if overlap_mask[i, j] > 0:
                    linearBlendingWithConstantWidth_img[i, j] = (
                        alpha_mask[i, j] * img_left[i, j] + 
                        (1 - alpha_mask[i, j]) * img_right[i, j]
                    )
        
        return linearBlendingWithConstantWidth_img
