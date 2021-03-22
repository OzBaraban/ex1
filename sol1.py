import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

"""both matrix values were copied from wikipedia"""
RGB_MAT = np.array([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]])
YIQ_MAT = np.array([[1.0, 0.956, 0.619], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])


def read_image(filename, representation):
    """
    reading the image
    :param filename - path to image:
    :param representation - int:
    :return picture in grayscale or rgb according to the input
    """
    im = imread(filename)
    if representation == 1:
        return rgb2gray(im)
    if representation == 2:
        im = im.astype(np.float64)
        im /= 256
        return im


def imdisplay(filename, representation):
    """
    display the image given in path using the previous function
    :param filename - path to image:
    :param representation - int:
    :return plotting the image in grayscale or rgb:
    """
    im = read_image(filename, representation)
    plt.imshow(im, cmap="gray")
    plt.show()


def rgb2yiq(imRGB):
    """
    transform the image format from RGB to YIQ
    :param imRGB vector:
    :return yiq vector:
    """
    return np.dot(imRGB, RGB_MAT.T)


def yiq2rgb(imYIQ):
    """
    transform the image format from YIQ to RGB
    :param imYIQ a vector:
    :return rgb a vector:
    """
    return np.dot(imYIQ, YIQ_MAT.T)


def histogram_equalize(im_orig):
    """
    equlize the histogram of a given image, and plots all the parameters needed.
    :param im_orig:
    :return im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).:
    """
    test_list = im_orig[0]
    test_list = list(test_list)
    if im_orig.ndim == 2:
        if test_list[0] > 1:
            im_orig /= 255
    im_mat = im_orig
    if im_orig.ndim == 3:                            #that means im_orig is a rgb picture.
        yiq_im = rgb2yiq(im_orig)
        im_mat = yiq_im[:, :, 0]                     #taking only the Y channel
    im_mat *= 255                                    #from [0,1] to [0, 255]
    im_mat = im_mat.astype(np.uint8)
    hist_orig = np.histogram(im_mat, 256)[0]         #histogram calculation via Numpy
    cum_hist = np.cumsum(hist_orig)                  #cumulative histogram calculation via Numpy
    num_of_pixels = cum_hist[255]
    cm = np.argmin(cum_hist[np.nonzero(cum_hist)])   # finding first non zero value in cum_hist
    t = ((cum_hist-cm)*255/(num_of_pixels-cm)).round().astype(np.uint8)
    new_y = t[im_mat]                                # new_y is a matrix representing the equalized image
    hist_eq = np.histogram(new_y, 256)[0]
    normal_new_y = new_y.astype(np.float64)/255
    if im_orig.ndim == 3:
        yiq_im[:, :, 0] = normal_new_y
        im_eq = yiq2rgb(yiq_im)                        #the equalied image in RGB format
        return [im_eq, hist_orig, hist_eq]
    im_eq = normal_new_y                               #the equalied image in GRAYSCALE format
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    this function quantize a given image and outputs it quanted state which defined by n_quant paramater and n_iter
    :param im_orig - a matrix which represent an image:
    :param n_quan - same as k in the error formula -  the number of intensities the quantedd image should have:
    :param n_iter -  the maximum number of iterations of the optimization procedure (may converge earlier):
    :return im_quant - is the quantized output image. (float64 image with values in [0, 1]).
            error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
            quantization procedure.:
    """
    counter = 1
    error_list = []
    """the following lines are the same as the ones at the start of histogram_equalize function because the setup 
        is the same"""
    imag_mat = im_orig
    y_dim = im_orig.ndim
    test_list = im_orig[0]
    test_list = list(test_list)
    if y_dim == 2:
        if test_list[0] > 1:
            im_orig /= 255
    if y_dim == 3:
        yiq_im = rgb2yiq(im_orig)
        imag_mat = yiq_im[:, :, 0]
    imag_mat *= 255
    imag_mat = imag_mat.astype(np.uint8)
    hist_orig = np.histogram(imag_mat, 256)[0]
    z = z_cal(hist_orig, n_quant)                                    #computes a good guess for z values
    curr_z = []
    while counter <= n_iter and curr_z != z:                         #n_iter iteration for minimizing the error and
                                                                     # convergence check
        curr_z = list(z)
        q = q_cal(z, n_quant, hist_orig)                             #computes optimal q values to minimize the error
        z = z_opt_calc(z, q, n_quant)                                #computes optimal z values to minimize the error
        error_temp = error_check(q, z, hist_orig, n_quant)           #the error given by current z's and q's
        error_list.append(error_temp)                                #creating the error list
        counter += 1
    error = np.array(error_list)
    final_q = np.array(np.round(q))
    q_map = np.zeros(256)
    for i in range(n_quant):                                          #creating the lut according to the q's and z's
        q_map[int(z[i]):int(z[i+1]+1)] = final_q[i]                   #q_map is the lut we're gonna work with
    y = q_map[imag_mat]                                               #y is a matrix representing the quantize image
    """same as histogram_equalize function end (after applying the lut on image)"""
    normal_new_y = y.astype(np.float64)/255
    if y_dim == 3:
        yiq_im[:, :, 0] = normal_new_y
        im_quant = yiq2rgb(yiq_im)
        return [im_quant, error]
    im_quant = normal_new_y
    return [im_quant, error]


def error_check(q, z, h, k):
    """
    this function calculate the error using the formula given in class err+= is equivalent to sum from 0 to k-1 due
    the for loop, the other sum is done by the numpy function sum, the parameters give to the function are named after
    their role in the mathematical formula
    :param q:
    :param z:
    :param h:
    :param k:
    :return int- the number which represent the error value :
    """
    err = 0
    for i in range(k):
        err += np.sum((q[i] - np.arange(int(z[i]), int(z[i+1]+1))) ** 2 * (h[int(z[i]):int(z[i + 1] + 1)]))
    return err


def z_cal(hist_orig, n_quant):
    """
    calcuate the initial z values to quantize with
    :param hist_orig - histogram of the image:
    :param n_quant - int: number of partition :
    :return z_list: a list of all the guessed z values :
    """
    z_list = [0]
    pixel_num = np.cumsum(hist_orig)[-1]
    pixel_num_for_z = pixel_num/n_quant                                           #calculate how much pixels per bin is optimal
    for i in range(1, n_quant):
        z_list.append(np.where(np.cumsum(hist_orig) >= pixel_num_for_z*i)[0][0])  #outputs the indices where the boolian
                                                                                  #is true
    z_list.append(255)                                                            #making sure the first and last z are 0 and
    return z_list                                                                 #and 255


def q_cal(z_list, n_quant, hist_orig):
    """
    calculate the q values according to the formula given in tirgul
    :param z_list:
    :param n_quant:
    :param hist_orig:
    :return q_list - a list of all the q values:
    # """
    q_list = []
    for i in range(n_quant):
        down = np.sum(hist_orig[int(z_list[i]):int(z_list[i + 1])+1])
        up = np.sum((np.arange(int(z_list[i]), int(z_list[i+1]+1)))*(hist_orig[int(z_list[i]):int(z_list[i+1])+1]))
        q = up/down
        q_list.append(q)
    return q_list


def z_opt_calc(z, q_list, n_quant):
    """
    calculate the z optimal according to the formula give in class
    :param z:
    :param q_list:
    :param n_quant:
    :return z - a list of z values after optimization:
    """
    z[0] = 0
    for i in range(n_quant-1):
        z[i+1] = int((q_list[i]+q_list[i+1])/2)
    z[-1] = 255
    return z
