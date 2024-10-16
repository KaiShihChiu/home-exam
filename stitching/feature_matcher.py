import matplotlib.pyplot as plt
import numpy as np
import cv2

class FeatureMatcher:
    def matchKeyPoint(self, kps_l, kps_r, features_l, features_r, ratio):
        '''
        Match keypoints between two images using a ratio test.

        Args:
            kps_l (list): Keypoints from the left image.
            kps_r (list): Keypoints from the right image.
            features_l (np.array): Feature descriptors from the left image.
            features_r (np.array): Feature descriptors from the right image.
            ratio (float): The ratio threshold for filtering matches as per Lowe's paper.

        Returns:
            list: List of matched keypoint positions [(point_in_left, point_in_right)].
        '''
        # Store the matching results: min and second min distances and their indices
        Match_idxAndDist = []
        
        # Loop over each feature in the left image
        for _, feature_l in enumerate(features_l):
            # Calculate Euclidean distances from the feature in the left image 
            # to all features in the right image
            dists = np.linalg.norm(feature_l - features_r, axis=1)
            
            # Get the index and value of the closest feature
            min_Idx = np.argmin(dists)
            min_dist = dists[min_Idx]
            
            # Exclude the closest feature and find the second closest
            dists[min_Idx] = np.inf  # Set the closest distance to infinity
            secMin_Idx = np.argmin(dists)
            secMin_dist = dists[secMin_Idx]

            # Store the matching result (index and distances of closest and second closest features)
            Match_idxAndDist.append([min_Idx, min_dist, secMin_Idx, secMin_dist])

        # Apply the ratio test to filter out weak matches
        goodMatches = [
            (i, m[0]) for i, m in enumerate(Match_idxAndDist) 
            if m[1] <= m[3] * ratio
        ]
        
        # Store the coordinates of good matches
        goodMatches_pos = []
        for (idx, correspondingIdx) in goodMatches:
            psA = (int(kps_l[idx].pt[0]), int(kps_l[idx].pt[1]))  # Left image keypoint
            psB = (int(kps_r[correspondingIdx].pt[0]), int(kps_r[correspondingIdx].pt[1]))  # Right image keypoint
            goodMatches_pos.append([psA, psB])
        
        return goodMatches_pos

    def drawMatches(self, imgs, matches_pos):
        '''
        Visualize matching keypoints between two images by drawing circles and lines connecting them.

        Args:
            imgs (tuple): A tuple of two images (left_image, right_image).
            matches_pos (list): A list of matched keypoint positions [(left_point, right_point)].

        Returns:
            np.array: A visualization image with matching points and connecting lines drawn.
        '''
        # Unpack the input images
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]  # Get dimensions of the left image
        (hr, wr) = img_right.shape[:2]  # Get dimensions of the right image

        # Initialize an empty visualization canvas large enough to hold both images side by side
        vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
        
        # Place the left and right images on the canvas
        vis[0:hl, 0:wl] = img_left  # Place the left image on the left side
        vis[0:hr, wl:] = img_right  # Place the right image on the right side
        
        # Draw the matching keypoints and lines connecting them
        for (img_left_pos, img_right_pos) in matches_pos:
            pos_l = img_left_pos  # Keypoint in the left image
            pos_r = (img_right_pos[0] + wl, img_right_pos[1])  # Keypoint in the right image (adjusted position)

            # Draw circles at the matching keypoints
            cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)  # Red circle for left image keypoint
            cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)  # Green circle for right image keypoint

            # Draw a blue line connecting the matching keypoints
            cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)  # Blue line between keypoints
        
        # Display the visualization image
        plt.figure()
        plt.title("Image with Matching Points")
        plt.imshow(vis[:, :, ::-1])  # Convert BGR to RGB for correct color display
        
        return vis
