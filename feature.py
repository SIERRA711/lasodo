import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

imagepath = input("Enter the path to the image: ")
# Load the image
image = cv2.imread(imagepath)
output_directory = os.path.dirname(imagepath)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def shi_tomasi(img, patch_size):
    """
    Computes the Shi-Tomasi corner response for a given image.

    :param img: 2D array representing the grayscale image.
    :type img: np.ndarray
    :param patch_size: Size of the patch (patch_size x patch_size) used for summation.
    :type patch_size: int
    :return: 2D array containing the Shi-Tomasi corner response for each pixel.
    :rtype: np.ndarray
    """
    
    # Define Sobel filters for gradient calculation in x and y directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Compute image gradients in x and y directions
    Ix = convolve2d(img, sobel_x, mode='valid')
    Iy = convolve2d(img, sobel_y, mode='valid')

    # Compute the products of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Create a summation filter based on the patch size
    summation_filter = np.ones((patch_size, patch_size))

    # Convolve the derivative products with the summation filter to obtain the elements of matrix M
    sum_Ixx = convolve2d(Ixx, summation_filter, mode='valid')
    sum_Ixy = convolve2d(Ixy, summation_filter, mode='valid')
    sum_Iyy = convolve2d(Iyy, summation_filter, mode='valid')

    # Calculate the trace and determinant of M
    trace = sum_Ixx + sum_Iyy
    determinant = sum_Ixx * sum_Iyy - sum_Ixy ** 2

    # Compute the Shi-Tomasi corner response (smallest eigenvalue of M)
    scores = (trace / 2) - np.sqrt((trace / 2) ** 2 - determinant)
    scores = np.maximum(scores, 0)  # Ensure non-negative scores

    # Add padding to match the size of the original image
    padding = patch_size // 2
    padded_scores = np.pad(scores, padding, mode='constant')

    return padded_scores

# Example usage
plt.figure()
score_image = shi_tomasi(gray_image, 9)
plt.imshow(score_image)
plt.show()
