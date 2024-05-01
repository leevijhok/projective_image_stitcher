"""
    Code estimating the shape of the target 
"""

import cv2
import numpy as np


def estimate_cylinder_dimensions(triangulated_points, scaling_factor=1):
    # Flatten the list of lists into a single list of 3D points
    points_flat = [point for sublist in triangulated_points for point in sublist]

    # Convert the flattened list to a NumPy array
    points = np.array(points_flat)

    # Determine the center of the cylinder
    center = np.mean(points, axis=0)

    # print("center", center)

    # Calculate radial distances from each point to the center
    radial_distances = np.linalg.norm(points - center, axis=1)

    # Estimate cylinder radius as the mean of radial distances
    cylinder_radius = np.max(radial_distances)
    # print("radial distances", radial_distances)
    # print("radius", cylinder_radius)

    return cylinder_radius * scaling_factor


def rectify_images_cylindrical(images, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K


    This function is a modified version of the function from:
    https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b

    """

    rectified_images = []

    for i in range(len(images)):

        # Image:
        img = images[i]

        # Image dimensions:
        height, width = img.shape[:2]

        # Pixel coordinates:
        v, u = np.indices((height, width))
        X = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(
            height * width, 3
        )  # to homog
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T  # normalized coords

        # Calculate cylindrical coords (sin\theta, h, cos\theta):
        Xc = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(
            width * height, 3
        )
        B = K.dot(Xc.T).T  # project back to image-pixels plane

        # Back from homog coords:
        B = B[:, :-1] / B[:, [-1]]

        # Make sure warp coords only within image bounds:
        B[(B[:, 0] < 0) | (B[:, 0] >= width) | (B[:, 1] < 0) | (B[:, 1] >= height)] = -1
        B = B.reshape(height, width, -1)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...

        # Warp the image according to cylindrical coords:
        rectified_image = cv2.remap(
            img_rgba,
            B[:, :, 0].astype(np.float32),
            B[:, :, 1].astype(np.float32),
            cv2.INTER_AREA,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        if rectified_image is None:
            print("Rectification failed for image", i)
            continue

        rectified_images.append(rectified_image)

    return rectified_images


def rectify_images_cylindrical_ba(images, K, rvecs, tvecs, r):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K



    This function is an even further modified version of the function from:
    https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b
    """

    rectified_images = []

    for i in range(len(images)):

        img = images[i]
        rvec = np.zeros((3, 1)) if i == 0 else rvecs[i - 1].reshape((3, 1))
        tvec = np.zeros((3, 1)) if i == 0 else tvecs[i - 1].reshape((3, 1))

        # Check if input image is valid
        if not isinstance(img, np.ndarray):
            print("Input image is not a valid NumPy array.")
            return None

        # Check if input image has correct dimensions
        if len(img.shape) != 3 or img.shape[2] != 3:
            print("Input image must be in BGR format.")
            return None

        # Image dimensions:
        height, width = img.shape[:2]

        # Pixel coordinates:
        v, u = np.indices((height, width))
        X = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(
            height * width, 3
        )  # to homog

        Kinv = np.linalg.inv(K)

        # Calculate cylindrical coords (sin\theta, h, cos\theta):
        X = Kinv.dot(X.T).T  # normalized coords
        Xc = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(
            width * height, 3
        )

        # Apply rotation and translation to transform to world coordinates
        R, _ = cv2.Rodrigues(rvec)
        Xc = Xc * r  # scale by radius
        Xc = np.dot(R, Xc.T).T + tvec.flatten()  # apply rotation and translation

        # Project back to image-pixels plane
        B = K.dot(Xc.T).T

        # Back from homog coords:
        B = B[:, :-1] / B[:, [-1]]

        # Make sure warp coords only within image bounds:
        B[(B[:, 0] < 0) | (B[:, 0] >= width) | (B[:, 1] < 0) | (B[:, 1] >= height)] = -1
        B = B.reshape(height, width, -1)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders

        # Warp the image according to cylindrical coords:
        rectified_image = cv2.remap(
            img_rgba,
            B[:, :, 0].astype(np.float32),
            B[:, :, 1].astype(np.float32),
            cv2.INTER_AREA,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        if rectified_image is None:
            print("Rectification failed for image", i)
            continue

        rectified_images.append(rectified_image)

    return rectified_images
