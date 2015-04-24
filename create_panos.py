import cv2
import pano_stitcher
import numpy
import os


def create_pano1():
    gray_left = cv2.imread("my_panos/images/left.png",
                           cv2.CV_LOAD_IMAGE_GRAYSCALE)
    left = cv2.imread("my_panos/images/left.png")  # , -1)
    gray_middle = cv2.imread("my_panos/images/middle.png",
                             cv2.CV_LOAD_IMAGE_GRAYSCALE)
    middle = cv2.imread("my_panos/images/middle.png")
    # ***middle = cv2.cvtColor(middle, cv2.COLOR_BGR2BGRA)
    gray_right = cv2.imread("my_panos/images/right.png",
                            cv2.CV_LOAD_IMAGE_GRAYSCALE)
    right = cv2.imread("my_panos/images/right.png")  # , -1)
    # rows, cols = left_image.shape

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow("image", left_image)
    # cv2.waitKey(0)

    # compute homography for left image
    homography1 = pano_stitcher.homography(gray_middle, gray_left)
    # warp left image
    warped_left, origin1 = pano_stitcher.warp_image(left, homography1)

    # compute homography for right image
    homography2 = pano_stitcher.homography(gray_middle, gray_right)
    # warp right image
    warped_right, origin2 = pano_stitcher.warp_image(right, homography2)

    # print("warped_left: ", warped_left.shape)
    # print("middle: ", middle.shape)
    # print("warped_right: ", warped_right.shape)

    images = (warped_left, warped_right, middle)
    origins = (origin1, origin2, (0, 0))
    mosaic1 = pano_stitcher.create_mosaic(images, origins)

    cv2.imwrite("feetMosaic.png", mosaic1)


create_pano1()
