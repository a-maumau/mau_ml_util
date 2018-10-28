import torch
import numpy as np
from PIL import Image

def torch_tensor_to_image(tensor, coeff=255):
    return numpy_to_image(tensor.detach().cpu().numpy(), coeff, transpose_channel=True)

def image_to_numpy(image):
    img_array = np.asarray(image)
    img_array.flags.writeable = True

    return img_array

def numpy_to_image(img_array, coeff=1, transpose_channel=False):
    if transpose_channel:
        return Image.fromarray(np.uint8(img_array.transpose(1,2,0)*coeff))
    else:
        return Image.fromarray(np.uint8(img_array*coeff))

def get_histogram(img_array, lower_bound=0, upper_bound=255, stride=10):
    """
        args:
            img_array: torch.Tensor, np.array or PIL.Image

            lower_bound: int (or float)
                lower bound of histogram.
                float is possible, but for floating point calculation error,
                sometimes it might not be what you expected.

            upper_bound: int (or float)
                upper bound of histogram.
                float is possible, but for floating point calculation error,
                sometimes it might not be what you expected.

            stride: int (or float)
                strides for grouping pixels.
                float is possible, but for floating point calculation error,
                sometimes it might not be what you expected.

                for instace, when stride = 10, then returning histogram will be like
                {"[0, 10)": 3, "[10, 20)": 31, "[20, 30)": 314, ...}

        return: dict
            returns a dictionary that key is a range and value is the count
    """

    histogram = {}

    if isinstance(img_array, torch.Tensor):
        stride_num = int((upper_bound-lower_bound)//stride)
        if ((upper_bound-lower_bound) % stride) == 0:
            stride_num += 1

        for i in range(stride_num):
            if stride*(i+1)+lower_bound >= upper_bound:
                histogram["[{}, {}]".format(stride*i+lower_bound, upper_bound)] = torch.sum(img_array[((stride*i+lower_bound) <= img_array) & (img_array < upper_bound)])
            else:
                histogram["[{}, {})".format(stride*i+lower_bound, stride*(i+1)+lower_bound)] = torch.sum(img_array[((stride*i+lower_bound) <= img_array) & (img_array < (stride*(i+1)+lower_bound))])

    elif isinstance(img_array, np.array):
        stride_num = int((upper_bound-lower_bound)//stride)
        if ((upper_bound-lower_bound) % stride) == 0:
            stride_num += 1

        for i in range(stride_num):
            if stride*(i+1)+lower_bound >= upper_bound:
                histogram["[{}, {}]".format(stride*i+lower_bound, upper_bound)] = torch.sum(img_array[((stride*i+lower_bound) <= img_array) & (img_array < upper_bound)])
            else:
                histogram["[{}, {})".format(stride*i+lower_bound, stride*(i+1)+lower_bound)] = torch.sum(img_array[((stride*i+lower_bound) <= img_array) & (img_array < (stride*(i+1)+lower_bound))])

    elif isinstance(img_array, Image.Image):
        if isinstance(lower_bound, int) and isinstance(upper_bound, int) and isinstance(stride, int):
            hist = ima_array.histogram()

            stride_num = int((upper_bound-lower_bound)//stride)
            if ((upper_bound-lower_bound) % stride) == 0:
                stride_num += 1

            for i in range(stride_num):
                if stride*(i+1)+lower_bound >= upper_bound:
                    histogram["[{}, {}]".format(stride*i+lower_bound, upper_bound)] = sum(hist[stride*i+lower_bound:])
                else:
                    histogram["[{}, {})".format(stride*i+lower_bound, stride*(i+1)+lower_bound)] = sum(hist[stride*i+lower_bound:stride*(i+1)+lower_bound])

    return histogram
