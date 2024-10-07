import os
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import nibabel as nib
from skimage import morphology, measure, segmentation


def preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, f"Failed to read image: {image_path}"
        # resize to the target size
        image = cv2.resize(image, target_size)
        # apply Gaussian blur for noise reduction
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # Normalizes the intensity of img
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        #Constrast limited adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) #convert back to RGB format

        return image, None
    except Exception as e:
        return None, f"Error processing {image_path}: {str(e)}"


# def skull_strip(image):
#     # Ensure the image is in float32 format
#     image = image.astype(np.float32)
#
#     # Normalize the image
#     image = (image - np.min(image)) / (np.max(image) - np.min(image))
#
#     # Apply Otsu's thresholding
#     thresh = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
#     # Perform morphological operations to clean up the mask
#     kernel = np.ones((5, 5), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#
#     # Find the largest connected component (which should be the brain)
#     labels = measure.label(opening)
#     props = measure.regionprops(labels)
#     largest_label = max(props, key=lambda prop: prop.area).label
#     brain_mask = labels == largest_label
#
#     # Fill any holes in the brain mask
#     brain_mask = morphology.remove_small_holes(brain_mask, area_threshold=64)
#
#     # Apply the mask to the original image
#     stripped_image = image * brain_mask
#
#     return stripped_image


# def preprocess_image(image_path, target_size=(224, 224)):
#     try:
#         # Load the NIfTI image
#         nii_img = nib.load(image_path)
#         image = nii_img.get_fdata()
#
#         # Assuming it's a 3D image, take a middle slice
#         middle_slice = image[:, :, image.shape[2] // 2]
#
#         # Perform skull stripping
#         stripped_image = skull_strip(middle_slice)
#
#         # Resize the image
#         stripped_image = cv2.resize(stripped_image, target_size)
#
#         # Apply additional preprocessing steps (as before)
#         image = cv2.GaussianBlur(stripped_image, (5, 5), 0)
#         image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         image = clahe.apply(image.astype(np.uint8))
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#
#         return image, None
#     except Exception as e:
#         return None, f"Error processing {image_path}: {str(e)}"


def process_and_validate_image(args):
    input_path, output_path, target_size = args
    try:
        processed_img, error = preprocess_image(input_path, target_size)
        if error:
            return False, error

        cv2.imwrite(output_path, processed_img)

        # Validate the processed image
        validated_img = cv2.imread(output_path)
        if validated_img.shape[:2] != target_size or validated_img.shape[2] != 3:
            return False, f"Validation failed for {output_path}"

        return True, None
    except Exception as e:
        return False, f"Error processing or validating {input_path}: {str(e)}"


def process_and_validate_dataset(input_base_path, output_base_path, target_size=(224, 224)):
    valid_count = 0
    invalid_count = 0
    invalid_files = []

    os.makedirs(output_base_path, exist_ok=True)

    tasks = []
    # nested loop structure traverse the dataset's directory structure
    for subset in ['Training', 'Testing']:
        for class_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            input_dir = os.path.join(input_base_path, subset, class_name)
            output_dir = os.path.join(output_base_path, subset, class_name)
            os.makedirs(output_dir, exist_ok=True)

            for img_name in os.listdir(input_dir):
                input_path = os.path.join(input_dir, img_name)
                output_path = os.path.join(output_dir, img_name)
                tasks.append((input_path, output_path, target_size))

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_and_validate_image, tasks), total=len(tasks),
                            desc="Processing and validating images"))

    for is_valid, error in results:
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            invalid_files.append(error)

    print(f"Valid images: {valid_count}")
    print(f"Invalid images: {invalid_count}")
    if invalid_files:
        print("Invalid files:")
        for file in invalid_files:
            print(file)

    return valid_count, invalid_count, invalid_files
#This structure allows for efficient processing of a large number of images:

if __name__ == '__main__':
    input_base_path = r'F:\Onedrive\Documents\GitHub\BrainScan-TL-MRI-Tumor-Classifier\brain-tumor-mri-dataset'
    output_base_path = r'F:\Onedrive\Documents\GitHub\BrainScan-TL-MRI-Tumor-Classifier\processed-dataset'

    valid_count, invalid_count, invalid_files = process_and_validate_dataset(input_base_path, output_base_path)