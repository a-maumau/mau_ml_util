"""
    wrapping the code in Pytorch
    to perform pair consistent of random values.
    I only wrapped the things I needed.
"""
import math
import random
import os
from PIL import Image, ImageOps
import numbers

def crop(img, i, j, h, w):
    if not isinstance(img, Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

def center_crop(img, output_size):
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        w, h = img.size
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        
        return crop(img, i, j, th, tw)

class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_img, target_img):
        for t in self.transforms:
            input_img, target_img = t(input_img, target_img)
        return input_img, target_img

class PairResize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, input_img, target_img):
        return input_img.resize((self.size, self.size), Image.ANTIALIAS), target_img.resize((self.size, self.size), Image.NEAREST)

class PairRandomHorizontalFlip(object):
    def __call__(self, input_img, target_img):
        if random.random() < 0.5:
            return input_img.transpose(Image.FLIP_LEFT_RIGHT), target_img.transpose(Image.FLIP_LEFT_RIGHT)
        return input_img, target_img

class PairCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input_img, target_img):
        return center_crop(input_img, self.size), center_crop(target_img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class PairRandomCrop(object):
    def __init__(self, size, padding=0):
        """
            size = (width, height)
        """
        
        if isinstance(size, numbers.Number):
            self.crop_sizew = int(size)
            self.crop_sizeh = int(size)
        else:
            self.crop_sizew = int(size[0])
            self.crop_sizeh = int(size[1])

        self.padding = padding

    def __call__(self, input_img, target_img):
        if self.padding > 0:
            input_img = ImageOps.expand(input_img, border=self.padding, fill=0)
            target_img = ImageOps.expand(target_img, border=self.padding, fill=0)

        # assuming input_img and target_img has same size
        w, h = input_img.size
        if w == self.crop_sizew and h == self.crop_sizeh:
            return input_img, target_img
        if w-self.crop_sizew < 0 or h-self.crop_sizeh < 0:
            add_size = w-self.crop_sizew if w-self.crop_sizeh < h-self.crop_sizeh else h-self.crop_sizeh
            input_img = input_img.resize((self.crop_sizew-add_size, self.crop_sizeh-add_size), Image.BILINEAR)
            target_img = target_img.resize((self.crop_sizew-add_size, self.crop_sizeh-add_size), Image.BILINEAR)
            w -= add_size
            h -= add_size

        x1 = random.randint(0, w - self.crop_sizew)
        y1 = random.randint(0, h - self.crop_sizeh)
        return input_img.crop((x1, y1, x1 + self.crop_sizew, y1 + self.crop_sizeh)), target_img.crop((x1, y1, x1 + self.crop_sizew, y1 + self.crop_sizeh))

class PairRandomSizedCrop(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input_img, target_img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            # assuming input_img and target_img has same sizw
            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                input_img = img.crop((x1, y1, x1 + w, y1 + h))
                target_img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(target_img.size == (w, h))
                assert(target_img.size == (w, h))

                return input_img.resize((self.size, self.size), self.interpolation), target_img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)

        return crop(scale(input_img)), crop(scale(target_img))

class PairRandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, input_img, target_img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return input_img.rotate(rotate_degree, Image.BILINEAR), target_img.rotate(rotate_degree, Image.NEAREST)
