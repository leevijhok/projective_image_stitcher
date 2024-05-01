import cv2
from cv2 import Stitcher
import numpy as np
import sys

sys.path.append("..")
from src.estimateFeatures import *


def stitchImagesFeatures(images):

    # Get keypoints:
    keypoints_list, descriptors_list = getSIFTKeypointsAndDescriptors(images=images)

    # Match Descriptors:
    matched_descriptors = matchDescriptors(
        descriptors_list=descriptors_list, distance_ratio_threshold=0.70, nMatches=50
    )

    # Matching of keypoints:
    points2d_list1, points2d_list2 = matchPoints(
        keypoints_list=keypoints_list, matched_descriptor_list=matched_descriptors
    )

    # Get homography:
    homographies = getHomographies(
        points2d_list1=points2d_list1, points2d_list2=points2d_list2
    )

    # Blend image and obtain the panorama:
    panorama = blendWarp(images=images, homographies=homographies)

    return panorama


def crop_black_areas(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find non-zero (non-black) pixels
    coords = cv2.findNonZero(gray)

    # Get the bounding box of the non-zero pixels
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image using the bounding box
    cropped_image = image[y : y + h, x : x + w]

    return cropped_image


def stitch_images_cv_iterative(images, batch_size=2):
    # Convert RGBA images to RGB or BGR format by discarding the alpha channel
    images_rgb = [
        cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) if image.shape[2] == 4 else image
        for image in images
    ]

    # Create a Stitcher object
    stitcher = cv2.Stitcher_create()

    # Disable OpenCL to avoid the error
    cv2.ocl.setUseOpenCL(False)

    # Initialize the stitched panorama
    panorama = images_rgb[0]

    # Start from the second image and iterate over the images in batches
    for i in range(1, len(images_rgb), batch_size):

        # Include the previous panorama in each batch
        batch_images = [panorama] + images_rgb[i : i + batch_size]

        # Stitch the images in the batch
        status, stitched_batch = stitcher.stitch(batch_images)

        # Check the stitching status
        if status == cv2.Stitcher_OK:
            # Update the panorama with the stitched batch
            panorama = crop_black_areas(stitched_batch)

            # Display the image in a window
            # cv2.imshow('Image', panorama)

            # Wait for a key press and close the window when any key is pressed
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            print(f"Stitched {len(batch_images)} images successfully!")
        elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Not enough images to perform stitching.")
            break  # Stop stitching if not enough images
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Homography estimation failed.")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Camera parameter adjustment failed.")
        else:
            print("Unknown error occurred during stitching. Status:", status)
            break  # Stop stitching on unknown error

    print(f"Stitched images successfully!")
    return panorama
