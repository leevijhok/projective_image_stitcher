import cv2
import numpy as np


def getSIFTKeypointsAndDescriptors(images: list):

    # Initialize SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)

    # List to store keypoints and descriptors of all images
    keypoints_list = []
    descriptors_list = []

    # Extract SIFT keypoints and descriptors for each image
    for img in images:

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply histogram equalization to improve feature detection:
        gray = cv2.equalizeHist(gray)

        # Apply Laplacian sharpening
        # blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        # gray = cv2.addWeighted(gray, 1.5, blurred, -1.0, 0)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Store keypoints
        keypoints_list.append(keypoints)

        # Check if descriptors are computed successfully
        if descriptors is not None and descriptors.ndim == 2:
            # L2 normalize descriptors
            descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-6
            descriptors_list.append(descriptors)
        else:
            print(
                f"Descriptors not computed successfully for image {len(keypoints_list) - 1}"
            )

    return keypoints_list, descriptors_list


def getORBKeypointsAndDescriptors(images):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # List to store keypoints and descriptors of all images
    keypoints_list = []
    descriptors_list = []

    # Extract ORB keypoints and descriptors for each image
    for img in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Apply sharpening to enhance edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(sharpened, None)

        # Store keypoints
        keypoints_list.append(keypoints)

        # Check if descriptors are computed successfully
        if descriptors is not None and descriptors.ndim == 2:
            descriptors_list.append(descriptors)
        else:
            print(
                f"Descriptors not computed successfully for image {len(keypoints_list) - 1}"
            )

    return keypoints_list, descriptors_list


def getBRISKKeypointsAndDescriptors(images):
    # Initialize BRISK detector
    brisk = cv2.BRISK_create()

    # List to store keypoints and descriptors of all images
    keypoints_list = []
    descriptors_list = []

    # Extract BRISK keypoints and descriptors for each image
    for img in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        # gray = cv2.equalizeHist(gray)

        # Apply Laplacian sharpening to enhance edges
        sharpened = cv2.Laplacian(gray, cv2.CV_64F)
        sharpened = np.uint8(np.absolute(sharpened))

        # Detect keypoints and compute descriptors
        keypoints, descriptors = brisk.detectAndCompute(sharpened, None)

        # Store keypoints
        keypoints_list.append(keypoints)

        # Check if descriptors are computed successfully
        if descriptors is not None and descriptors.ndim == 2:
            descriptors_list.append(descriptors)
        else:
            print(
                f"Descriptors not computed successfully for image {len(keypoints_list) - 1}"
            )

    return keypoints_list, descriptors_list


def getSURFKeypointsAndDescriptors(images):
    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create()

    # List to store keypoints and descriptors of all images
    keypoints_list = []
    descriptors_list = []

    # Extract SURF keypoints and descriptors for each image
    for img in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = surf.detectAndCompute(gray, None)

        # Store keypoints
        keypoints_list.append(keypoints)

        # Check if descriptors are computed successfully
        if descriptors is not None and descriptors.ndim == 2:
            descriptors_list.append(descriptors)
        else:
            print(
                f"Descriptors not computed successfully for image {len(keypoints_list) - 1}"
            )

    return keypoints_list, descriptors_list


def getFASTKeypointsAndDescriptors(images):
    # Initialize FAST detector
    fast = cv2.FastFeatureDetector_create()

    # List to store keypoints and descriptors of all images
    keypoints_list = []
    descriptors_list = []

    # Extract FAST keypoints and descriptors for each image
    for img in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect keypoints using FAST
        keypoints = fast.detect(blurred, None)

        # Compute descriptors (FAST doesn't provide descriptors, so use a placeholder)
        descriptors = np.zeros((len(keypoints), 64), dtype=np.uint8)

        # Store keypoints and descriptors
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


def getAKAZEKeypointsAndDescriptors(images):
    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create()

    # List to store keypoints and descriptors of all images
    keypoints_list = []
    descriptors_list = []

    # Extract AKAZE keypoints and descriptors for each image
    for img in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Apply bilateral filtering to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = akaze.detectAndCompute(filtered, None)

        # Store keypoints
        keypoints_list.append(keypoints)

        # Check if descriptors are computed successfully
        if descriptors is not None and descriptors.ndim == 2:
            descriptors_list.append(descriptors)
        else:
            print(
                f"Descriptors not computed successfully for image {len(keypoints_list) - 1}"
            )

    return keypoints_list, descriptors_list


def matchDescriptors(descriptors_list, distance_ratio_threshold=0.80, nMatches=100):
    matched_descriptor_list = []

    for i in range(len(descriptors_list) - 1):

        # Brute-force matching:
        bf = cv2.BFMatcher(
            cv2.NORM_L2, crossCheck=False
        )  # Disable crossCheck for ratio test

        # Finding the nearest neighbors:
        matches = bf.knnMatch(
            descriptors_list[i], descriptors_list[i + 1], k=2
        )  # k=2 for ratio test

        if matches is None:
            print(f"No matches found between descriptors {i} and {i+1}")
            matched_descriptor_list.append([])
            continue

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < distance_ratio_threshold * n.distance:
                good_matches.append(m)

        # Sort matches based on distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Select top-n matches
        good_matches = good_matches[:nMatches]

        if len(good_matches) < 8:
            print(f"{i}: Not enough matches to triangulation")

        matched_descriptor_list.append(good_matches)

    return matched_descriptor_list


def matchPoints(keypoints_list, matched_descriptor_list):

    points2d_list1 = []
    points2d_list2 = []

    for i in range(len(matched_descriptor_list)):

        matches = matched_descriptor_list[i]
        keypoints1 = keypoints_list[i]
        keypoints2 = keypoints_list[i + 1]

        if len(matches) < 4:
            print("Error: Insufficient matches for triangulation")

        # Image points
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Append the projected 2D points to the list
        points2d_list1.append(pts1)
        points2d_list2.append(pts2)

    return points2d_list1, points2d_list2


def getHomographies(points2d_list1, points2d_list2):

    homographies = []

    for pts1, pts2 in zip(points2d_list1, points2d_list2):
        # Append homographies:
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        homographies.append(H)

    return homographies


def blendWarp(images, homographies):

    # Final output panorama:
    panorama = None

    for i in range(len(homographies)):
        image1 = images[i]
        image2 = images[i + 1]
        H = homographies[i]

        # Warp the first image using the homography
        result = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))

        # Check if it's the first iteration, if so, initialize panorama with the first warped image
        if panorama is None:
            panorama = result
        else:
            # Blending the warped image with the current panorama using alpha blending
            alpha = 0.1  # blending factor
            panorama = cv2.addWeighted(result, alpha, panorama, 1 - alpha, 0)

        # Display the image in a window
        # cv2.imshow('Image', panorama)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return panorama
