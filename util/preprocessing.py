import cv2
import numpy as np
from torchvision import transforms
import torch

# def merged_img(warp1,warp2,mask1,mask2):
#     alpha = 0.5
#     beta = 1 - alpha
#     overlap_mask = cv2.bitwise_and(mask1, mask2)
#     roi = cv2.addWeighted(warp1, alpha, warp2, beta, 0)
#     img2_masked = warp2 & cv2.bitwise_not(overlap_mask)[:,:,None]
#     merged = cv2.add(warp1, img2_masked, roi)
#     return merged



def merged_img(warp1, warp2, mask1, mask2):
    # Check if masks and images have the same size
    if warp1.shape[:2] != mask1.shape[:2] or warp2.shape[:2] != mask2.shape[:2]:
        print("Warning: Image and mask dimensions do not match.")
        return None

    alpha = 0.5
    beta = 1 - alpha

    # Ensure binary masks (0 or 255)
    _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    # Create the overlap mask (common region of mask1 and mask2)
    overlap_mask = cv2.bitwise_and(mask1, mask2)

    # Apply the overlap mask to both warp1 and warp2
    img1_masked = cv2.bitwise_and(warp1, warp1, mask=cv2.bitwise_not(overlap_mask))
    img2_masked = cv2.bitwise_and(warp2, warp2, mask=overlap_mask)

    # Blend the two images with specified weights
    blended_img = cv2.addWeighted(img1_masked, alpha, img2_masked, beta, 0)

    return blended_img

def preprocess_image(image,newsize):
    image = image.convert("RGB")
    image = transforms.Resize(newsize)(image)
    return image


def preprocess_map(map,newsize):
    map = map.convert("L")
    map = transforms.Resize(newsize)(map)
    map = transforms.ToTensor()(map)
    return map

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image