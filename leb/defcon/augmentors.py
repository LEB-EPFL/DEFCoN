# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics
# See the LICENSE.txt file for more details.

import numpy as np

def adjust_limits(image):
    """Corrects the limits of the image or the stack of images after
    augmentation to stay in the [0,1] range

    """
    immin, immax = 0, 1
    image[image < immin] = immin
    image[image > immax] = immax
    return image

def brightness(image):
    """Randomly augments a batch by increasing or decreasing the brightness

    """
    bright = np.random.uniform(low=-0.5, high=1.0, size=(image.shape[0],1,1,1))
    image *= 10**bright
    image = adjust_limits(image)
    return image

def contrast(stack):
    """Randomly augments a stack of 3D images by changing the contrast
    (shifting the values away from the mean, or towards it)

    """
    center = (stack.max(axis=(1,2,3), keepdims=True) - stack.min(axis=(1,2,3), keepdims=True))*0.5+ stack.min(axis=(1,2,3), keepdims=True)
    mult_contrast = np.random.uniform(low=1.0, high=4.0, size=(stack.shape[0],1,1,1))
    stack -= center
    stack *= mult_contrast
    stack += center*mult_contrast
    stack = adjust_limits(stack)
    return stack

def gaussnoise(stack):
    """Adds a random amount of noise on each image of a stack of 3D images

    """
    noise_level = np.random.uniform(low=-5.0, high=-3.0, size=stack.shape[0])
    for i, image in enumerate(stack):
        mean = 0
        var = 10**noise_level[i]
        noise = np.random.normal(mean, var, size=image.shape)
        image += noise
        image = adjust_limits(image)
        stack[i] = image
    return stack

def intensity_curves(stack):
    """Randomly adjusts the intensity curves on a stack of 3D images

    """
    min_int = np.random.uniform(low=0.0, high=0.05, size=(stack.shape[0],1,1,1))
    max_int = np.random.uniform(low=0.7, high=1.2, size=(stack.shape[0],1,1,1))
    stack *= max_int
    stack += min_int
    stack = adjust_limits(stack)
    return stack


#%%
def random_crops(X, y, y_seg, n_crops=10, crop_dim=None):
    """Randomly cut crops in the input images X and the corresponding crops in
    the target images y

    Parameters
    ----
    X : np.array() with shape=(n_images, x_pixels, y_pixels, 1)
        SMLM image stack
    y : np.array() with shape=(n_images, x_pixels, y_pixels, 1)
        Density map stack of X
    y_seg : np.array() with shape=(n_images, x_pixels, y_pixels, 1)
        Segmentation mask stack of X
    n_crops : int
        Number of crops to cut per image (default 10)
    crop_dim : int
        length (in pixels) of the square crops to cut from the images. If None
        (default), crop_dim is a random int divisible by 4 between 16 and the
        height of the original image

    Returns
    ----
    (X_new, y_new): tuple of np.array() with
    shape=(n_crops*n_images, crop_dim, crop_dim, 1)
        The two stacks containing the crops

    """


    max_dim = int(X.shape[1]/4 - 1)

    img_dim1 = X.shape[1]
    img_dim2 = X.shape[2]
    nImages = X.shape[0]

    X_new = np.array([])
    y_new = np.array([])
    y_seg_new = np.array([])
    for i in range(n_crops):
        #Choose the size of the crop
        if crop_dim == None:
            crop_dim = 4*np.random.randint(4, max_dim)
        # Resize X_new, y_seg_new and y_new
        X_new.resize((X_new.shape[0] + nImages, crop_dim, crop_dim, 1))
        y_new.resize((y_new.shape[0] + nImages, crop_dim, crop_dim, 1))
        y_seg_new.resize((y_new.shape[0] + nImages, crop_dim, crop_dim, 1))
        # Draw crops
        ind1 = np.random.randint(low=0, high=img_dim1-crop_dim, size=(nImages,1))
        ind2 = np.random.randint(low=0, high=img_dim2-crop_dim, size=(nImages,1))

        (grid, img_ind) = np.meshgrid(np.arange(crop_dim), np.arange(nImages))
        ind1 = ind1 + grid
        ind2 = ind2 + grid

        X_temp = X[img_ind, ind1, :, :]
        X_new[-nImages:] = X_temp[img_ind, :, ind2, :]
        y_temp = y[img_ind, ind1, :, :]
        y_new[-nImages:] = y_temp[img_ind, :, ind2, :]
        y_seg_temp = y_seg[img_ind, ind1, :, :]
        y_seg_new[-nImages:] = y_seg_temp[img_ind, :, ind2, :]

    return (X_new, y_new, y_seg_new)