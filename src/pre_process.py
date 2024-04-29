"""

Renames the target images into the proper format.

"""


import os
import argparse
import cv2
from PIL import Image

def rename_files(src_folder):
    # Get list of files in the folder
    files = os.listdir(src_folder)

    # Sort files alphabetically
    files.sort()

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
            os.rename(os.path.join(src_folder, file), os.path.join(src_folder, new_name))
            
            # Increment counter
            count += 1

def lower_resolution_in_folder(folder_path, output_folder, scale_percent):
    """
    Lower the resolution of all images in the given folder and save them to the output folder.
    Update the resolution metadata of JPEG images.

    Parameters:
        folder_path (str): Path to the folder containing images.
        output_folder (str): Path to the folder where resized images will be saved.
        scale_percent (int): Percentage by which to scale down the images (e.g., 50 for 50% reduction).

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Loop through each file in the folder
    for file_name in file_list:
        # Check if the file is an image (ending with .jpg, .png, etc.)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            # Read the image
            img = cv2.imread(os.path.join(folder_path, file_name))

            # Get the new dimensions based on the scale percentage
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            # Resize the image
            resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, resized_img)

            # Update metadata for JPEG images
            if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                update_jpeg_metadata(output_path, width, height)

            print(f"Image '{file_name}' resized and saved to '{output_path}'.")

def update_jpeg_metadata(image_path, width, height):
    """
    Update the resolution metadata of a JPEG image.

    Parameters:
        image_path (str): Path to the JPEG image.
        width (int): New width of the image.
        height (int): New height of the image.

    Returns:
        None
    """
    try:
        # Open the image using Pillow
        img = Image.open(image_path)

        # Update the image metadata
        img.info["dpi"] = (300, 300)  # Example DPI value, adjust as needed
        img.info["jfif_density"] = (300, 300)  # Example density value, adjust as needed

        # Save the updated image
        img.save(image_path, dpi=(300, 300))

        print(f"Metadata updated for image '{image_path}'.")
    except Exception as e:
        print(f"Error updating metadata for image '{image_path}': {e}")


def pre_process_all(src_folder, 
                trg_folder, 
                src_folder2, 
                trg_folder2, 
                filetype, 
                lower_resolution=False):
    
    #rename_files(args.src_folder)
    lower_resolution_in_folder(src_folder, trg_folder, 30)
    lower_resolution_in_folder(src_folder2, trg_folder2, 30)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-processess the images.')
    parser.add_argument('-src_folder', type=str, default="..\data\raw\demo_1_high", help='The name of the source folder.')
    parser.add_argument('-trg_folder', type=str, default="..\data\processed\demo_1", help='The name of the target folder.')
    parser.add_argument('-src_folder2', type=str, default="...\data\raw\Chessboard_high", help='The name of the source folder.')
    parser.add_argument('-trg_folder2', type=str, default="..\data\processed\Chessboard", help='The name of the target folder.')
    parser.add_argument('-filetype', type=str, default="png", help='The target image file-type.')
    parser.add_argument('-lower_resolution', type=str, default=False, help='Lowers the resolution of input images.')
    args = parser.parse_args()
    rename_files(args.src_folder)
    lower_resolution_in_folder(args.src_folder, args.trg_folder, 30)
    lower_resolution_in_folder(args.src_folder2, args.trg_folder2, 30)
