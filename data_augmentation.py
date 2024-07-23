import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

# Define augmentation pipeline
augmentation_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flip with 50% probability
    # iaa.Flipud(0.5),  # vertical flip with 50% probability
    iaa.Affine(
        rotate=(-45, 45),  # rotate between -45 and 45 degrees
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # pan by -20 to +20 percent
    ),
    iaa.PiecewiseAffine(scale=(0.01, 0.05))  # distortions
], random_order=True)

def read_annotations(file_path):
    with open(file_path, 'r') as file:
        annotations = file.readlines()
    return annotations

def write_annotations(file_path, annotations):
    with open(file_path, 'w') as file :
        file.writelines(annotations)

def parse_annotation(annotation):
    data = annotation.strip().split()
    class_index = int(data[0])
    coords = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
    return class_index, coords

def format_annotation(class_index, coords):
    coords = coords.flatten()
    annotation = f"{class_index} " + " ".join(map(str, coords)) + "\n"
    return annotation

def normalize_coordinates(coords, image_shape):
    h, w = image_shape[:2]
    coords = coords.astype(np.float32)
    coords[:, 0] /= w
    coords[:, 1] /= h
    return coords

def denormalize_coordinates(coords, image_shape):
    h, w = image_shape[:2]
    coords = coords.astype(np.float32)
    coords[:, 0] *= w
    coords[:, 1] *= h
    return coords

def augment_image_and_annotations(image, annotations):
    keypoints = []
    class_indices = []
    for annotation in annotations:
        class_index, coords = parse_annotation(annotation)
        coords = denormalize_coordinates(coords, image.shape)
        keypoints.append(ia.KeypointsOnImage([ia.Keypoint(x=coord[0], y=coord[1]) for coord in coords], shape=image.shape))
        class_indices.append(class_index)

    image_aug, keypoints_aug = augmentation_pipeline(image=image, keypoints=keypoints)
    annotations_aug = []
    for i, keypoint in enumerate(keypoints_aug):
        coords_aug = np.array([(kp.x, kp.y) for kp in keypoint.keypoints])
        coords_aug = normalize_coordinates(coords_aug, image_aug.shape)
        annotation_aug = format_annotation(class_indices[i], coords_aug)
        annotations_aug.append(annotation_aug)
    
    return image_aug, annotations_aug

input_image_folder = "images"
input_annotation_folder = "annotations"
output_image_folder = "aug_of_ann"
output_annotation_folder = "aug_of_img"

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_annotation_folder, exist_ok=True)

for image_name in os.listdir(input_image_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_image_folder, image_name)
        annotation_path = os.path.join(input_annotation_folder, image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        image = np.array(Image.open(image_path))
        annotations = read_annotations(annotation_path)
        
        image_aug, annotations_aug = augment_image_and_annotations(image, annotations)
        
        output_image_path = os.path.join(output_image_folder, image_name)
        output_annotation_path = os.path.join(output_annotation_folder, image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        
        Image.fromarray(image_aug).save(output_image_path)
        write_annotations(output_annotation_path, annotations_aug)
