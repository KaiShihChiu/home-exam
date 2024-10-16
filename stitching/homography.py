import numpy as np
from sklearn.linear_model import RANSACRegressor

class Homography:
    def solve_homography(self, P, m):
        """
        Solve the homography matrix using point correspondences.
        
        Args:
            P (np.array): Source points (Nx2).
            m (np.array): Destination points (Nx2).
        
        Returns:
            np.array: The 3x3 homography matrix.
        """
        # Initialize the A matrix for the homogeneous linear system
        A = np.zeros((2 * len(P), 9))
        
        # Construct the A matrix based on source and destination points
        for r in range(len(P)):
            A[2 * r] = [-P[r, 0], -P[r, 1], -1, 0, 0, 0, P[r, 0] * m[r, 0], P[r, 1] * m[r, 0], m[r, 0]]
            A[2 * r + 1] = [0, 0, 0, -P[r, 0], -P[r, 1], -1, P[r, 0] * m[r, 1], P[r, 1] * m[r, 1], m[r, 1]]
        
        # Solve for the homography matrix using SVD (Singular Value Decomposition)
        _, _, vt = np.linalg.svd(A)
        H = np.reshape(vt[-1], (3, 3))
        
        # Normalize the homography matrix to ensure H[2, 2] = 1
        H /= H[2, 2]
        
        return H
    
    def fitHomoMat(self, matches_pos, threshold=5.0, num_iter=8000):
        """
        Fit a homography matrix using RANSAC (Random Sample Consensus) to robustly
        find the transformation between two sets of points.
        
        Args:
            matches_pos (list of tuples): A list of matched point pairs, each as (destination point, source point).
            threshold (float, optional): Inlier distance threshold for RANSAC. Default is 5.0.
            num_iter (int, optional): Number of RANSAC iterations. Default is 8000.
        
        Returns:
            np.array: The best-fit homography matrix.
        """
        # Convert matched positions to NumPy arrays
        dstPoints = np.array([list(dstPoint) for dstPoint, _ in matches_pos])
        srcPoints = np.array([list(srcPoint) for _, srcPoint in matches_pos])
        
        # Number of matched points
        NumSample = len(matches_pos)
        
        # Track the maximum number of inliers and the best homography matrix found
        MaxInlier = 0
        Best_H = None
        
        # Convert source points to homogeneous coordinates (append a column of ones)
        srcPoints_homogeneous = np.hstack([srcPoints, np.ones((srcPoints.shape[0], 1))])
        
        # RANSAC loop: iterate for the specified number of iterations
        for _ in range(num_iter):
            # Randomly sample 4 points to compute a homography
            SubSampleIdx = np.random.choice(NumSample, 4, replace=False)
            
            # Solve homography based on the 4 randomly chosen correspondences
            H = self.solve_homography(srcPoints[SubSampleIdx], dstPoints[SubSampleIdx])
            
            # Transform all source points using the current homography
            dst_transformed = (H @ srcPoints_homogeneous.T).T
            dst_transformed /= dst_transformed[:, 2][:, np.newaxis]  # Normalize by the last coordinate
            
            # Compute Euclidean distances between the transformed points and actual destination points
            distances = np.linalg.norm(dst_transformed[:, :2] - dstPoints, axis=1)
            
            # Count inliers: points whose distance is less than the specified threshold
            inliers = np.sum(distances < threshold)
            
            # Update the best homography if more inliers are found
            if MaxInlier < inliers:
                MaxInlier = inliers
                Best_H = H
        
        # Return the best homography matrix found
        return Best_H
