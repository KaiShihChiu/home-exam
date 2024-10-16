
# **Image Background**
1. Three images were taken with a moving camera.
2. Each image contains two sizes of coins, with some of them overlapping.
3. The background intensity and object intensity may vary across images.
4. The purpose is to simulate periodic structures and possible overlaps of known structures in real-world images.

---

# **Goals**
1. Stitch all images. 
2. Semantic segmentation to separate coins with different sizes.  
3. Remove outliers (if existed).  
4. Detect overlapping coins.

---

# **Run (Root Directory)**
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the **stitching process**:
   ```bash
   python stitch.py
   ```
   **A.** Input: 3 images.  
   **B.** Output: original images, the matched keypoints, and panorama images.
   
3. Run the **segmentation process**:
   ```bash
   python seg.py
   ```
   **A.** Input: a panorama image.  
   **B.** Output: semantic segmentation image, the segmentation image with overlap and non-overlap contours.

---

# **Code Structures**
```
img/
    IMG_7113.jpg
    IMG_7114.jpg
    IMG_7115.jpg

segmentation/
    __init__.py
    circle_detection.py
    clustering.py
    contour_analysis.py
    image_processing.py
    segmentation.py
    visualization.py

stitching/
    __init__.py
    blending.py
    feature_detector.py
    feature_matcher.py
    homography.py
    stitcher.py
    utils.py

requirements.txt
seg.py
stitch.py
```

---

# **Stitching Package**
- **`utils.py`**:
  - Contains utility functions used across modules, such as image resizing, cropping, or removing black borders.
  
- **`feature_detector.py`**:
  - Detects keypoints and features from input images using SIFT.

- **`feature_matcher.py`**:
  - Matches features between images using Lowe’s ratio test.
  - Visualizes matches by drawing lines and circles between matching points.

- **`homography.py`**:
  - Calculates the homography matrix to align and warp images for stitching.
  - Uses RANSAC to fit the best transformation model.

- **`stitcher.py`**:
  - Contains the main stitching logic that integrates feature matching, homography, and blending.
  - Handles the entire stitching pipeline by iteratively combining multiple images.

- **`blending.py`**:
  - Implements blending algorithms to combine overlapping image regions smoothly.
  - Includes linear blending and blending with constant widths.

---

# **Segmentation Package**
- **`image_processing.py`**:
  - Contains preprocessing operations like filtering, thresholding, and edge detection.
  - Prepares images for segmentation by removing noise and enhancing features.

- **`contour_analysis.py`**:
  - Extracts and analyzes contours found in the images.
  - Computes area and perimeter as features for K-means clustering.

- **`clustering.py`**:
  - Handles K-means clustering with scores to automatically determine the optimal number of clusters.
  - Includes logic for outlier detection and RANSAC-based outlier removal.

- **`segmentation.py`**:
  - Implements the main segmentation logic to separate different objects in the image.
  - Uses contour-based segmentation or pixel classification techniques.

- **`circle_detection.py`**:
  - Detects circles using the Hough Circle Transform.
  - Validates circles to ensure they fall within specific segmented regions.

- **`visualization.py`** (optional):
  - Provides functions to visualize segmentation results using Matplotlib or OpenCV.
  - Displays clustered segments, labeled regions, or segmentation masks.

