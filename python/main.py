# Jason Kayser
# jpk2f2

import cv2  # import opencv
import histogram_equalization as he  # import personal histogram equalization algorithm

# import images
image_a = cv2.imread('resources/Fig0308(a)(fractured_spine).tif')
image_a2 = cv2.imread('resources/Fig0308(a)(fractured_spine).tif', 0)  # read in grayscale
image_b = cv2.imread('resources/Fig0309(a)(washed_out_aerial_image).tif')
image_b2 = cv2.imread('resources/Fig0309(a)(washed_out_aerial_image).tif', 0)  # read in grayscale
image_c = cv2.imread('resources/Fig0316(2)(2nd_from_top).tif')
image_c2 = cv2.imread('resources/Fig0316(2)(2nd_from_top).tif', 0)  # read in grayscale

he.equalize(image_a, image_a2)  # equalize and display image 308
he.equalize(image_b, image_b2)  # equalize and display image 309
he.equalize(image_c, image_c2)  # equalize and display image 316(2)
