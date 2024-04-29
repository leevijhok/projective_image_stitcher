
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

def rename_files(folderName, files):

    # Counter for new file names
    count = 1

    # Iterate over files and rename them
    for file in files:
        # Get file extension
        _, extension = os.path.splitext(file)

        # Check if the file is an image (you may want to adjust this check based on your file types)
        if extension.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
            # Construct new file name with leading zeros
            new_name = f"{count:03d}{extension}"
            
            # Rename the file
            os.rename(os.path.join(folderName, file), os.path.join(folderName, new_name))
            
            # Increment counter
            count += 1

def get_images(folderName, pre_process = False):

    images = []

    # Get list of files in the folder
    files = os.listdir(folderName)

    # Sort files alphabetically
    files.sort()

    for filename in files:
        img_path = os.path.join(folderName, filename)
        img = cv2.imread(img_path)

        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {img_path}")

    image_shape = (images[0].shape[0], images[0].shape[1])

        
    return images, image_shape


def displayImages(img1, img2):

    # Create a figure and axis object
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))  # 2 rows, 2 columns

    # Plot each image on its corresponding axis
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Non-warped panorama')

    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Cylindrically warped panorama with bundle adjustment')

    plt.tight_layout()
    plt.show()


def saveImage(file_path, image):
    # Save the image to the specified file path
    cv2.imwrite(file_path, image)