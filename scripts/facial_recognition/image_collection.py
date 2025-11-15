"""
Image Collection Module
Load and display facial images for each team member.
"""

import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        ndarray: Image data in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image


def display_images(images_dict, title="Facial Images"):
    """
    Display multiple images in a grid.
    
    Args:
        images_dict (dict): Dictionary with image names as keys and image arrays as values
        title (str): Title for the plot
    """
    num_images = len(images_dict)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for idx, (name, image) in enumerate(images_dict.items()):
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(image_rgb)
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def load_member_images(member_id, raw_images_path):
    """
    Load all images for a specific member.
    
    Args:
        member_id (str): Member identifier (e.g., 'member1')
        raw_images_path (str): Path to raw images directory
        
    Returns:
        dict: Dictionary with image types as keys and image arrays as values
    """
    member_path = os.path.join(raw_images_path, member_id)
    
    if not os.path.exists(member_path):
        raise FileNotFoundError(f"Member directory not found: {member_path}")
    
    images = {}
    for filename in os.listdir(member_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(member_path, filename)
            images[filename] = load_image(image_path)
    
    return images


if __name__ == "__main__":
    # Example usage
    raw_path = "../../data/raw/images"
    
    # Load and display member1 images
    member1_images = load_member_images("member1", raw_path)
    display_images(member1_images, "Member 1 - Facial Images")
