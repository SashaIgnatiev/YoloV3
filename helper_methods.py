import cv2
import numpy as np

# TODO: reimplement this later with a better implementation avoiding the use of opencv
def resize_image(img, shape = (448, 448)):
    '''
    Resize the image to given shape.
    :param shape: desired shape of image
    :param img: original image
    :return: resized image
    '''

    old_shape = img.shape[:2]
    new_shape = shape

    #scale ratio
    r = min(new_shape[0] / old_shape[0], new_shape[1] / old_shape[1])
    new_unpad = int(round(old_shape[1] * r)), int(round(old_shape[0] * r))

    # Resize the image
    im = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    #padding
    color = (114, 114, 114)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    # compute padding on all corners
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im


def LeakyReLU(x, alpha = 0.1):
    '''
    Implementation of the Leaky ReLU activation function.
    :param x: input tensor (we are going to apply the activation function element wise)
    :param alpha: chosen slope for values below 0
    :return: non-linear applied activation function to each element
    '''
    if x > 0:
        return x
    else:
        return alpha * x