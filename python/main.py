# Jason Kayser
# jpk2f2

import cv2  # import opencv
import numpy
import matplotlib.pyplot as plot

image_a = cv2.imread('resources/Fig0308(a)(fractured_spine).tif')
image_a2 = cv2.imread('resources/Fig0308(a)(fractured_spine).tif', 0)  # read in grayscale
image_b = cv2.imread('resources/Fig0309(a)(washed_out_aerial_image).tif')
image_b2 = cv2.imread('resources/Fig0309(a)(washed_out_aerial_image).tif', 0)  # read in grayscale
image_c = cv2.imread('resources/Fig0316(2)(2nd_from_top).tif')
image_c2 = cv2.imread('resources/Fig0316(2)(2nd_from_top).tif', 0)  # read in grayscale


# helps keep from repeating code unnecessarily
def test_func(im1, im2):
    # resize images in order to stack them later
    image = cv2.resize(im1, (0, 0), None, .70, .70)
    image2 = cv2.resize(im2, (0, 0), None, .70, .70)
    image2 = image2.astype(numpy.uint8)  # fix type

    equ = cv2.equalizeHist(image2)  # create perfect example

    image_prop = image.shape  # get image properties
    height, width, channels = image_prop
    #  print(height, width, channels)
    total = height * width

    flatimage = image.flatten()  # flatten to 1D
    plot.hist(flatimage, bins=256)  # create histogram
    plot.xlim(0, 256)  # prevent cutoff
    plot.title('first')
    plot.show()

    # person 'histogram' function for counting pixel usage
    # was printing this, but pyplot.hist() makes prettier ones and I couldn't get it to match
    def _histogram(image, bins):
        array = numpy.zeros(bins)  # initialize zeroed array of bin-size(always 256)
        for i in image:
            array[i] += 1
        return array

    histogram = _histogram(flatimage, 256)  # use personal histogram function to get array

    # plot.plot(histogram)
    # plot.show()

    csum = 0  # holds current cumulative sum
    cumusum = numpy.zeros(len(histogram))  # array for holding cumulative sum steps

    # get cumulative sum
    for i in range(0, len(histogram)):
        csum = csum + histogram[i]
        cumusum[i] = csum

    # plot cumulative sum
    plot.plot(cumusum)
    plot.title('cumusum')
    plot.show()
    # based off of https://en.wikipedia.org/wiki/Histogram_equalization#Implementation
    # normalize cumulative sum
    num = (cumusum - cumusum.min()) * 255
    denom = cumusum.max() - cumusum.min()
    cumusum = num / denom

    # plot normalized cumulative sum
    plot.plot(cumusum)
    plot.title('cumusum normalized')
    plot.show()

    cumusum = cumusum.astype('uint8')  # fix type

    img_new = cumusum[flatimage]  # create final image

    # histogram = _histogram(img_new, bins=256)
    plot.hist(img_new, bins=256)  #plot final image
    plot.title('final histogram')
    plot.show()

    img_new = numpy.reshape(img_new, image.shape)  # reshape dimensions

    equ_3_channel = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)  # reshape channels so dimensions match
    res = numpy.hstack((image, equ_3_channel, img_new))  # concatenate images for display
    cv2.imshow('test', res)  # show stacked images
    cv2.waitKey(0)  # wait for button press to close window


test_func(image_a, image_a2)  # image 308
test_func(image_b, image_b2)  # image 309
test_func(image_c, image_c2)  # image 316(2)
