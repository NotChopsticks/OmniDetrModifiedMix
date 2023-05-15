import os
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.io import read_image, write_jpeg
from torchvision.transforms.functional import convert_image_dtype


def mixup_images(image1, image2, alpha=0.5):
    mixed_image = alpha * image1 + (1 - alpha) * image2
    return mixed_image


def main(aerial_folder, ground_folder, output_folder, alpha=0.5):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)


    aerial_images = sorted(os.listdir(aerial_folder))
    ground_images = sorted(os.listdir(ground_folder))

    lam = beta.sample([len(aerial_images)]) #.to(device="cuda")
    lam = torch.max(lam, 1. - lam)
    # lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))

    os.makedirs(output_folder, exist_ok=True)

    for aerial_image_name, ground_image_name, l in zip(aerial_images, ground_images, lam):
        aerial_image_path = os.path.join(aerial_folder, aerial_image_name)
        ground_image_path = os.path.join(ground_folder, ground_image_name)

        aerial_image = convert_image_dtype(read_image(aerial_image_path), dtype=torch.float32)
        ground_image = convert_image_dtype(read_image(ground_image_path), dtype=torch.float32)

        if aerial_image.shape != ground_image.shape:
            print(f"Skipping {aerial_image_name} and {ground_image_name} due to different shapes.")
            continue

        mixed_image = mixup_images(aerial_image, ground_image, l)
        mixed_image = convert_image_dtype(mixed_image, dtype=torch.uint8)
        output_path = os.path.join(output_folder, f"mixed_{aerial_image_name}")
        write_jpeg(mixed_image, output_path)


if __name__ == "__main__":
    aerial_data = "D:/General_Storage/SDU-GAMODv4/SDU-GAMODv4/scaled_dataset/val/droneview"
    aerial_annotation = "D:/General_Storage/SDU-GAMODv4/SDU-GAMODv4/supervised_annotations/aerial/aligned_ids/aerial_valid_aligned_ids_w_perspective.json"
    ground_data = "D:/General_Storage/SDU-GAMODv4/SDU-GAMODv4/scaled_dataset/val/groundview"
    ground_annotation = "D:/General_Storage/SDU-GAMODv4/SDU-GAMODv4/supervised_annotations/ground/aligned_ids/ground_valid_aligned_ids_w_perspective.json"

    aerial_folder = aerial_data
    ground_folder = ground_data
    output_folder = "D:/General_Storage/SDU-GAMODv4/SDU-GAMODv4/scaled_dataset/val/new_mixup"
    alpha = (10, 2)  # MixUp interpolation coefficient (0.0 <= alpha <= 1.0)

    main(aerial_folder, ground_folder, output_folder, alpha)
