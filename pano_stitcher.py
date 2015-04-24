"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def homography(image_a, image_b):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(image_a, None)
    kp2, des2 = sift.detectAndCompute(image_b, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if (m.distance / n.distance) < 0.7:
            good.append(m)

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = image_a.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    return M


def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """

    p1 = np.ones(3, np.float32)
    p2 = np.ones(3, np.float32)
    p3 = np.ones(3, np.float32)
    p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    p1[:2] = [0, 0]
    p2[:2] = [x, 0]
    p3[:2] = [0, y]
    p4[:2] = [x, y]

    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for pt in [p1, p2, p3, p4]:
        hp = np.dot(np.matrix(homography, np.float32),
                    np.matrix(pt, np.float32).T)

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([[hp_arr[0] / hp_arr[2]],
                             hp_arr[1] / hp_arr[2]], np.float32)

        if(max_x is None or normal_pt[0, 0] > max_x):
            max_x = normal_pt[0, 0]

        if(max_y is None or normal_pt[1, 0] > max_y):
            max_y = normal_pt[1, 0]

        if(min_x is None or normal_pt[0, 0] < min_x):
            min_x = normal_pt[0, 0]

        if(min_y is None or normal_pt[1, 0] < min_y):
            min_y = normal_pt[1, 0]

    translationMatrix = np.zeros(shape=(3, 3))
    translationMatrix[0] = [0, 0, -min_x]
    translationMatrix[1] = [0, 0, -min_y]
    newHomography = np.add(homography, translationMatrix)

    warp = cv2.warpPerspective(image,
                               newHomography,
                               (int(max_x - min_x), int(max_y - min_y)))

    return warp, (int(min_x), int(min_y))


def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.

    Arguments:
      images: a list of 4-channel images to combine in the mosaic.
      origins: a list of the locations upper-left corner of each image in
               a common frame, e.g. the frame of a central image.

    Returns: a new 4-channel mosaic combining all of the input images. pixels
             in the mosaic not covered by any input image should have their
             alpha channel set to zero.
    """
    # This will make the first image, in the images list,
    # have an origin of (0,0)
    # so that we can stitch them sequentially.
    new_origins = []
    o_x, o_y = origins[0]
    for origin in origins:
        x, y = origin
        new_origins.append([x + abs(o_x), y + abs(o_y)])

    # Dimensions for the mosaic
    max_height = 0
    max_width = 0
    for image, origin in zip(images, new_origins):
        x, y = image.shape[:2]
        max_width += origin[0]
        max_height += origin[1]

    max_width += images[1].shape[1]
    max_height += images[1].shape[0]

    final = np.ones((max_height, max_width, images[0].shape[2]), np.uint8)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2BGRA)

    for i in range(len(images)):
        y, x, _ = images[i].shape
        o_x, o_y = new_origins[i]

        final[abs(o_y):abs(y) + abs(o_y),
              abs(o_x):abs(x) + abs(o_x), :_] = images[i]

    # cv2.imshow('final', final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return final
