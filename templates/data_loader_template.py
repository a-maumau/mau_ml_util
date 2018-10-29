from ..utils.path_util import path_join, list_dir

import pickle
from PIL import Image

import torch
import torch.utils.data as data

class Template_ClassificationDatasetLoader(data.Dataset):
    def __init__(self, img_root, img_list_path, dataset_pickle_path=None,
                       input_transform=None,
                       load_all_in_ram=True, img_ext=".jpg", img_convert_type="RGB",
                       pickle_img_key="image", pickle_label_key="label",
                       pickle_path_data=False, pickle_path_relative=False,
                       return_original=False):
        """
            args:
                img_root: str
                    root directory of input images.

                img_list_path: str
                    path to a file which is written a image name.
                    if this is "not" None, it will only use this image written in this file.
                    it is considered to be style of <img_name> <class id> like
                        img_001 3
                        img_002 1
                        img_003 4
                        .
                        .
                        .
                      in the file. which is name class id
                    if this is None, it will read all file in the img_root directory.
                      in this scenario, if you set load_all_in_ram=False, it might raise some
                      errors if there is a non opneable file with PIL in the directory or no pairs.
                    setting the option of img_exr, or mask_ext to use different extensions.

                dataset_pickle_path: str
                    path to a pickled dataset.
                    if this is None, it will not use pickled dataset.
                    the dataset must be a list of dictionary, which is like
                        [ {"image_name":<img_name>, "image_label":<label>}, ...]
                    label must be int

                input_transform: function
                    function that compose transform to PIL.Image object for image.
                    torchvision.transforms is considered as a typical function.
                    if it is None, transforms.ToTensor will only be performed.

                load_all_in_ram: bool
                    if this is True. the all dataset image will be loaded on the memory.
                    if you cause no memory problem, you can set this to False,
                    and this loader will only load the file paths at the initial moment.

                img_ext: str
                    extension for image.
                    it will not used in loading from pickle.
                    if there is multiple extensions, use pickled datase you can preprocessed.

                img_convert_type: str
                    open a image in mode of this.
                    for detail of modes, see the PIL docmentation.

                pickle_img_key: str
                    key of input image in dictionary of the pickle dataset.

                pickle_label_key: str
                    key of label in dictionary of the pickle dataset.

                pickle_path_data: bool
                    if this is True, it will open the image based on the pickled data.
                    if this is False, it will directly map to self.imgs and mask_imgs.

                pickle_path_relative: bool
                    if this is True, it will open the image in pickled data as relative path depends on
                    arguments, img_root and mask_root.
                    if this is False, it will open the image in pickled data considering
                    as a absolute path.
        """

        self.input_transform = input_transform
        self.load_all_in_ram = load_all_in_ram
        self.img_convert_type = img_convert_type
        self.return_original = return_original

        self.dataset = []

        # read from folders
        if dataset_pickle_path is None:
            with open(path_join(img_list_path), "r") as file:
                image_list = file.readlines()

            data_list = [img_name.split(" ") for img_name in image_list]

            # label is str
            for (img_name, label) in data_list:
                try:
                    if load_all_in_ram:
                        img = Image.open(path_join(img_root, img_name+img_ext)).convert(self.img_convert_type)
                    else:
                        img = path_join(img_root, img_name+img_ext)

                    self.dataset.append({"image":img, "label":int(label)})

                except Exception as e:
                    print(e)
                    print("pass {}".format(img_name))

        # read from pickled data
        else:
            with open(dataset_pickle_path, "rb") as f:
                pickled_dataset = pickle.load(f)

            if pickle_path_data:
                if self.load_all_in_ram:
                    try:
                        if pickle_path_relative:
                            self.dataset = list(map(lambda x: {"image":Image.open(path_join(img_root, x[pickle_img_key])).convert(self.img_convert_type),
                                                               "label":x[pickle_label_key].convert("P")},
                                                    pickled_dataset)
                                               )
                        else:
                            self.dataset = list(map(lambda x: {"image":Image.open(x[pickle_img_key]).convert(self.img_convert_type),
                                                               "label":x[pickle_label_key].convert("P")},
                                                    pickled_dataset)
                                               )

                    except Exception as e:
                        print(e)
                        print("cannot load image.")
                else:
                    if pickle_path_relative:
                        self.dataset = list(map(lambda x: {"image":path_join(img_root, x[pickle_img_key]),
                                                           "label":x[pickle_label_key]},
                                                pickled_dataset)
                                           )
                    else:
                        self.dataset = pickled_dataset

            else:
                self.dataset = pickled_dataset
        
        self.data_num = len(self.dataset)
    
    def __getitem__(self, index):
        if self.load_all_in_ram:
            img = self.dataset[index]["image"]
            label = self.dataset[index]["label"]
        else:
            img = Image.open(self.dataset[index]["image"]).convert(self.img_convert_type)
            label = self.dataset[index]["label"]

        if self.return_original:
            original_img = img.copy()
                
        if self.input_transform is not None:
            img = self.input_transform(img)
        else:
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor)
            else:
                img = torch.from_numpy(img.transpose(2,0,1)).type(torch.FloatTensor)

        if self.return_original:
            # output of the loader in the batch, label is just a tuple so you need torch.LongTensor()
            return img, label, torch.from_numpy(np.asarray(original_img)).type(torch.LongTensor)

        return img, label

    def __len__(self):
        return self.data_num

class Template_SegmentationDatasetLoader(data.Dataset):
    def __init__(self, img_root, mask_root, img_list_path=None, dataset_pickle_path=None,
                       pair_transform=None, input_transform=None, target_transform=None,
                       load_all_in_ram=True, img_ext=".jpg", mask_ext=".png", img_convert_type="RGB",
                       pickle_img_key="image", pickle_mask_key="mask",
                       pickle_path_data=False, pickle_path_relative=False,
                       return_original=False):
        """
            args:
                img_root: str
                    root directory of input images.

                mask_root: str
                    root directory of mask images.

                img_list_path: str
                    path to a file which is written a image name.
                    if this is "not" None, it will only use this image written in this file.
                    it is considered to be like
                        img_001
                        img_002
                        img_003
                        .
                        .
                        .
                      in the file.
                    if this is None, it will read all file in the img_root directory.
                      in this scenario, if you set load_all_in_ram=False, it might raise some
                      errors if there is a non opneable file with PIL in the directory or no pairs.
                    setting the option of img_exr, or mask_ext to use different extensions.

                dataset_pickle_path: str
                    path to a pickled dataset.
                    if this is None, it will not use pickled dataset.
                    the dataset must be a list of dictionary, which is like
                        [ "key for input image":<input image>, "key for GT mask":<mask image>}, ...]

                pair_transform: function
                    function that compose transform to PIL.Image object for image and mask.
                    this function must take 2 PIL.Image object which is (image, mask).
                    if it is None, nothing will be done.

                input_transform: function
                    function that compose transform to PIL.Image object for image.
                    torchvision.transforms is considered as a typical function.
                    if it is None, transforms.ToTensor will only be performed.

                target_transform: function
                    function that compose transform to PIL.Image object for mask.
                    torchvision.transforms is considered as a typical function.
                    if it is None, it will convert to torch.LongTensor.

                load_all_in_ram: bool
                    if this is True. the all dataset image will be loaded on the memory.
                    if you cause no memory problem, you can set this to False,
                    and this loader will only load the file paths at the initial moment.

                img_ext: str
                    extension for image.
                    it will not used in loading from pickle.
                    if there is multiple extensions, use pickled datase you can preprocessed.

                mask_ext: str
                    extension for mask image.
                    it will not used in loading from pickle.
                    if there is multiple extensions, use pickled datase you can preprocessed.

                img_convert_type: str
                    open a image in mode of this.
                    for detail of modes, see the PIL docmentation.

                pickle_img_key: str
                    key of input image in dictionary of the pickle dataset.

                pickle_mask_key: str
                    key of ground truth mask in dictionary of the pickle dataset.

                pickle_path_data: bool
                    if this is True, it will open the image based on the pickled data.
                    if this is False, it will directly map to self.imgs and mask_imgs.

                pickle_path_relative: bool
                    if this is True, it will open the image in pickled data as relative path depends on
                    arguments, img_root and mask_root.
                    if this is False, it will open the image in pickled data considering
                    as a absolute path.
        """

        self.pair_transform = pair_transform
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.load_all_in_ram = load_all_in_ram
        self.img_convert_type = img_convert_type
        self.return_original = return_original

        self.dataset = []

        # read from folders
        if dataset_pickle_path is None:
            if img_list_path is None:
                input_name_list = []
                mask_name_list = []

                input_image_list = list_dir(path_join(img_root))
                for name in input_image_list:
                    # replacing both extensions is for when img_root and mask_root are same
                    name_list.append(name.replace(img_ext, "").replace(mask_ext, ""))

                mask_image_list = list_dir(path_join(mask_root))
                for name in mask_image_list:
                    # replacing both extensions is for when img_root and mask_root are same
                    mask_name_list.append(name.replace(img_ext, "").replace(mask_ext, ""))

                # to erase duplication of names
                image_list = list(set(input_name_list+mask_name_list))
            else:
                with open(path_join(img_list_path), "r") as file:
                    image_list = file.readlines()
                    image_list = [img_name.rstrip("\n") for img_name in image_list]

            for img_name in image_list:
                try:
                    if self.load_all_in_ram:
                        img = Image.open(path_join(img_root, img_name+img_ext)).convert(self.img_convert_type)
                        mask_img = Image.open(path_join(mask_root, img_name+mask_ext)).convert('P')
                    else:
                        img = path_join(img_root, img_name+img_ext)
                        mask_img = path_join(mask_root, img_name+mask_ext)

                    self.dataset.append({"image":img, "mask":mask_img})

                except Exception as e:
                    print(e)
                    print("pass {}".format(img_name))

        # read from pickled data
        else:
            with open(dataset_pickle_path, "rb") as f:
                pickled_dataset = pickle.load(f)

            if pickle_path_data:
                if self.load_all_in_ram:
                    try:
                        if pickle_path_relative:
                            self.dataset = list(map(lambda x: {"image":Image.open(path_join(img_root, x[pickle_img_key])).convert(self.img_convert_type),
                                                                "mask":Image.open(path_join(img_root, x[pickle_mask_key])).convert("P")},
                                                    pickled_dataset)
                                                )
                        else:
                            self.dataset = list(map(lambda x: {"image":Image.open(x[pickle_img_key]).convert(self.img_convert_type),
                                                                "mask":Image.open(x[pickle_mask_key]).convert("p")},
                                                    pickled_dataset)
                                                )

                        self.dataset.append({"image":img, "mask":mask_img})

                    except Exception as e:
                        print(e)
                        print("cannot load image.")
                else:
                    if pickle_path_relative:
                        self.dataset = list(map(lambda x: {"image":path_join(img_root, x[pickle_img_key]),
                                                            "mask":path_join(img_root, x[pickle_mask_key])}, pickled_dataset)
                                            )
                    else:
                        self.dataset = pickled_dataset
            else:
                self.dataset = pickled_dataset
        
        self.data_num = len(self.dataset)
    
    def __getitem__(self, index):
        if self.load_all_in_ram:
            img = self.dataset[index]["image"]
            mask = self.dataset[index]["mask"]
        else:
            img = Image.open(self.dataset[index]["image"]).convert(self.img_convert_type)
            mask = Image.open(self.dataset[index]["mask"]).convert('P')

        if self.pair_transform is not None:
            img, mask = self.pair_transform(img, mask)

        if self.return_original:
            original_img = img.copy()
                
        if self.input_transform is not None:
            img = self.input_transform(img)
        else:
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor)
            else:
                img = torch.from_numpy(img.transpose(2,0,1)).type(torch.FloatTensor)
                
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.asarray(mask)).type(torch.LongTensor)

        if self.return_original:
            return img, mask, torch.from_numpy(np.asarray(original_img)).type(torch.LongTensor)

        return img, mask

    def __len__(self):
        return self.data_num
