import os
import cv2
import numpy as np

# Function to crop the brain image by finding the first white pixel from each side and saving the crop dimensions
def crop_brain_image(image, margin=2):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to binary (white pixels are set to 255, black to 0)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find the first and last non-black pixel in each dimension
    rows = np.any(binary_image, axis=1)
    cols = np.any(binary_image, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add margin
    y_min = max(0, y_min - margin)
    y_max = min(binary_image.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(binary_image.shape[1], x_max + margin)

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, (y_min, y_max, x_min, x_max)

# Function to crop the mask image using saved dimensions
def crop_mask_image(image, crop_dims):
    y_min, y_max, x_min, x_max = crop_dims
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

# Function to resize image to 512x512
def resize_image(image, size=(512, 512)):
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image

# Main processing function for brain images and segment masks
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        if "_processed_image.png" in image_name:
            # Brain image processing
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            # Crop and resize brain image
            cropped_image, crop_dims = crop_brain_image(image)
            resized_image = resize_image(cropped_image)

            # Save the processed brain image
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, resized_image)

            # Segment mask processing
            mask_name = image_name.replace("_processed_image", "_processed_mask")
            mask_path = os.path.join(input_folder, mask_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path)
                cropped_mask = crop_mask_image(mask, crop_dims)
                resized_mask = resize_image(cropped_mask)

                # Save the processed mask
                mask_output_path = os.path.join(output_folder, mask_name)
                cv2.imwrite(mask_output_path, resized_mask)

# Define paths
input_folder = 'test'         # Folder containing the original images and masks
output_folder = 'test_cropped'     # Folder where processed images and masks will be saved

# Process images (crop and resize)
process_images(input_folder, output_folder)

#print("Image processing complete. Check the output folder for cropped and resized images and masks.")