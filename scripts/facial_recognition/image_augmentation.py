"""
Image Augmentation Module
Apply various augmentation techniques to facial images.
"""

import cv2
import numpy as np
from albumentations import Compose, Rotate, HorizontalFlip, GaussNoise, Brightness


def rotate_image(image, angle=15):
    """
    Rotate image by specified angle.
    
    Args:
        image (ndarray): Input image
        angle (int): Rotation angle in degrees
        
    Returns:
        ndarray: Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated


def flip_image(image, axis=1):
    """
    Flip image horizontally or vertically.
    
    Args:
        image (ndarray): Input image
        axis (int): 0 for vertical flip, 1 for horizontal flip
        
    Returns:
        ndarray: Flipped image
    """
    return cv2.flip(image, axis)


def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image (ndarray): Input image in BGR format
        
    Returns:
        ndarray: Grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adjust_brightness(image, brightness_factor=0.7):
    """
    Adjust image brightness.
    
    Args:
        image (ndarray): Input image
        brightness_factor (float): Factor to adjust brightness (0.5 = darker, 1.5 = brighter)
        
    Returns:
        ndarray: Brightness-adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_noise(image, mean=0, std=15):
    """
    Add Gaussian noise to image.
    
    Args:
        image (ndarray): Input image
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise
        
    Returns:
        ndarray: Image with noise added
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def apply_augmentations(image, augmentation_list):
    """
    Apply multiple augmentations to an image.
    
    Args:
        image (ndarray): Input image
        augmentation_list (list): List of augmentation functions to apply
        
    Returns:
        list: List of augmented images
    """
    augmented_images = [image]  # Include original
    
    for aug_func in augmentation_list:
        augmented_images.append(aug_func(image))
    
    return augmented_images


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from image_collection import load_image
    
    # Load sample image
    sample_image = load_image("../../data/raw/images/member1/neutral.jpg")
    
    # Apply augmentations
    augmentations = [
        lambda img: rotate_image(img, 15),
        lambda img: flip_image(img),
        lambda img: convert_to_grayscale(img),
        lambda img: adjust_brightness(img, 1.3)
    ]
    
    augmented = apply_augmentations(sample_image, augmentations)
    
    # Display results
    fig, axes = plt.subplots(1, len(augmented), figsize=(15, 5))
    for idx, img in enumerate(augmented):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()
