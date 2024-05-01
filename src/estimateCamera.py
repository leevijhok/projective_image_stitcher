import cv2
import numpy as np
from scipy.optimize import minimize


def calibrate_camera(images, boardShape=(8, 6)):
    # Prepare object points, considering the size of squares
    objp = np.zeros((boardShape[0] * boardShape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : boardShape[0], 0 : boardShape[1]].T.reshape(-1, 2)

    # Rest of your code remains unchanged...
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    for img in images:
        # Convert image to BGR if it's not already in that format
        if len(img.shape) == 2:  # Check if the image is grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, boardShape, None)

        # If corners are found, add object points, image points
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
            imgpoints.append(corners2)

    # Check if the number of object points and image points are equal
    if len(objpoints) != len(imgpoints) or len(objpoints) == 0:
        raise ValueError(
            "Number of object points and image points do not match or no valid corners found"
        )

    # Perform camera calibration
    ret, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return K, dist_coeff


def estimate_projection_matrices(points2d_list1, points2d_list2, K):

    P_list1 = []
    P_list2 = []

    P_prev = np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(points2d_list1)):
        # Get matched keypoints
        src_pts = points2d_list1[i]
        dst_pts = points2d_list2[i]

        # Estimate fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 0.1, 0.99)

        # Select only inlier points
        src_pts = src_pts[mask.ravel() == 1]
        dst_pts = dst_pts[mask.ravel() == 1]

        # Compute essential matrix from fundamental matrix and camera intrinsic matrix
        E = np.dot(K.T, np.dot(F, K))

        # Decompose essential matrix to get rotation and translation
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)

        # Construct projection matrices
        P1 = P_prev  # Assume first camera is at origin
        P2 = np.hstack((R, t))

        # Enforce the right-hand rule
        if np.linalg.det(P2[:, :3]) < 0:
            P2 = -P2

        P_list1.append(P1)
        P_list2.append(P2)

        P_prev = P2

    return P_list1, P_list2


def rectifyCamera(images, K, dist_coeffs):
    """
    Rectify the distortion caused by camera
    """
    return [cv2.undistort(image, K, dist_coeffs) for image in images]


def straighten_image(image, K, dist_coeffs):
    """
    Rectify the final panorama
    """

    # Get image dimensions
    height, width = image.shape[:2]

    # Compute the rectification transformation
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, dist_coeffs, (width, height), 1, (width, height)
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K, (width, height), cv2.CV_32FC1
    )
    rectified_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return rectified_image
