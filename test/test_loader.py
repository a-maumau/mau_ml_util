from ..utils.path_util import path_join, list_dir

import pickle
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

class Test_Dataset(data.Dataset):
    """
        this class only returns the image, no label, no mask.
        used for wether data augmentation is correct or not.
    """
    def __init__(self, img_root, input_transform=None, input_normalize=None,
                       img_ext=".jpg", return_original=False):
        """
            args:
                img_root: str
                    root directory of input images.

                input_transform: function
                    function that compose transform to PIL.Image object for image.
                    torchvision.transforms is considered as a typical function.
                    if it is None, transforms.ToTensor will only be performed.

                input_normalize: function
                    if you want get the non normalized image, you should set the
                    normalizing part of input_transform to this.
                    `return_original` option will return the image after input_transform.

                img_ext: str
                    extension for image.
                    it filter the images containing in `img_root` by extension.

                return_original: bool
                    if this is True, it will returns the non-normalized image,
                    which `input_transform` is applied.
        """

        self.input_transform = input_transform
        self.input_normalize = input_normalize
        self.return_original = return_original
        self.dataset = []

        # take only img_ext matched image.
        data_list = [img_name for img_name in list_dir(img_root) if img_ext in img_name]

        # label is str
        for img_name in data_list:
            try:
                img = Image.open(path_join(img_root, img_name+img_ext)).convert("RGB")
                self.dataset.append(img)

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))
        
        self.data_num = len(self.dataset)
    
    def __getitem__(self, index):
        img = self.dataset[index]

        if self.input_transform is not None:
            img = self.input_transform(img)
        else:
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor)
            else:
                img = torch.from_numpy(img.transpose(2,0,1)).type(torch.FloatTensor)

        if self.return_original:
            if isinstance(_img, torch.Tensor):
                original_img = img.clone()
            else:
                original_img = torch.from_numpy(np.asarray(img.copy())).type(torch.FloatTensor)

        if self.input_normalize is not None:
            img = self.input_normalize(img)

        if self.return_original:
            return img, original_img

        return img

    def __len__(self):
        return self.data_num

class Test_ClassificationDataset(data.Dataset):
    """
        this class returns the image and random binary label.
    """
    def __init__(self, img_root, input_transform=None, input_normalize=None,
                       img_ext=".jpg", return_original=False):
        """
            args:
                img_root: str
                    root directory of input images.

                input_transform: function
                    function that compose transform to PIL.Image object for image.
                    torchvision.transforms is considered as a typical function.
                    if it is None, transforms.ToTensor will only be performed.

                img_ext: str
                    extension for image.
                    it will not used in loading from pickle.
                    if there is multiple extensions, use pickled datase you can preprocessed.

                input_normalize: function
                    if you want get the non normalized image, you should set the
                    normalizing part of input_transform to this.
                    `return_original` option will return the image after input_transform.

                img_ext: str
                    extension for image.
                    it filter the images containing in `img_root` by extension.

                return_original: bool
                    if this is True, it will returns the non-normalized image.
        """

        self.input_transform = input_transform
        self.input_normalize = input_normalize
        self.return_original = return_original
        self.dataset = []

        # take only img_ext matched image.
        data_list = [img_name for img_name in list_dir(img_root) if img_ext in img_name]

        # label is str
        for img_name in data_list:
            try:
                img = Image.open(path_join(img_root, img_name+img_ext)).convert(self.img_convert_type)

                # I did not prepare a label, so sample from random 
                self.dataset.append({"image":img, "label":np.random.random_integers(1)})

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))
        
        self.data_num = len(self.dataset)
    
    def __getitem__(self, index):
        img = self.dataset[index]["image"]
        label = self.dataset[index]["label"]
                
        if self.input_transform is not None:
            img = self.input_transform(img)
        else:
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor)
            else:
                img = torch.from_numpy(img.transpose(2,0,1)).type(torch.FloatTensor)

        if self.return_original:
            if isinstance(_img, torch.Tensor):
                original_img = img.clone()
            else:
                original_img = torch.from_numpy(np.asarray(img.copy())).type(torch.FloatTensor)

        if self.input_normalize is not None:
            img = self.input_normalize(img)

        if self.return_original:
            return img, label,original_img

        return img, label

    def __len__(self):
        return self.data_num

class Test_SegmentationDataset(data.Dataset):
    """
        this class returns the image and random binary mask.
    """
    def __init__(self, img_root, pair_transform=None, input_transform=None, input_normalize=None, target_transform=None,
                       img_ext=".jpg", return_original=False, img_size=(255, 255)):
        """
            args:
                img_root: str
                    root directory of input images.

                pair_transform: function
                    function that compose transform to PIL.Image object for image.
                    this transform will be applied to input image and target mask.
                    mau_ml_util.transform.pair_transforms is considered as a typical function.

                input_transform: function
                    function that compose transform to PIL.Image object for image.
                    torchvision.transforms is considered as a typical function.
                    if it is None, transforms.ToTensor will only be performed.

                input_normalize: function
                    if you want get the non normalized image, you should set the
                    normalizing part of input_transform to this.
                    `return_original` option will return the image after input_transform.

                target_transform: function
                    this is transform for a target image.
                    usually, it will be not used, which means None.

                img_ext: str
                    extension for image.
                    it filter the images containing in `img_root` by extension.

                return_original: bool
                    if this is True, it will returns the non-normalized image.

                img_size: (int, int)
                    it is used for making the random mask.
        """

        self.pair_transform = pair_transform
        self.input_transform = input_transform
        self.input_normalize = input_normalize
        self.target_transform = target_transform
        self.return_original = return_original
        self.dataset = []

        # take only img_ext matched image.
        data_list = [img_name for img_name in list_dir(img_root) if img_ext in img_name]

        # label is str
        for img_name in data_list:
            try:
                img = Image.open(path_join(img_root, img_name+img_ext)).convert(self.img_convert_type)
                # I did not prepare a mask, so sample from random 
                mask = Image.fromarray(numpy.uint8(randint(0, 2, img_size)))
                
                self.dataset.append({"image":img, "mask":mask})

            except Exception as e:
                print(e)
                print("pass {}".format(img_name))
        
        self.data_num = len(self.dataset)
    
    def __getitem__(self, index):
        img = self.dataset[index]["image"]
        mask = self.dataset[index]["mask"]

        # pair ###############################
        if self.pair_transform is not None:
            img, mask_img = self.pair_transform(img, mask)
        else:
            img = img
            mask = mask
        # ------------------------------------
        
        # input image ########################
        if self.input_transform is not None:
            img = self.input_transform(img)
        else:
            img = torch.from_numpy(np.asarray(img).transpose(2,0,1)).type(torch.FloatTensor)

        if self.return_original:
            if isinstance(_img, torch.Tensor):
                original_img = img.clone()
            else:
                original_img = torch.from_numpy(np.asarray(img.copy())).type(torch.FloatTensor)

        if self.input_normalize is not None:
            img = self.input_normalize(img)
        # ------------------------------------

        # target image #######################
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.asarray(mask)).type(torch.LongTensor)
        # ------------------------------------

        if self.return_original:
            return img, mask, original_img

        return img, mask

    def __len__(self):
        return self.data_num
