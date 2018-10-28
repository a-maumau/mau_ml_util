from ..templates.template_data_loader import Template_SegmentationTestLoader

class SegmentationTestLoader(Template_SegmentationTestLoader):
    def __init__(self, img_root, input_transform=None):
        self.input_transform = input_transform

        self.img_root = img_root

        self.image_names = os.listdir(os.path.join(img_root))

        self.data_num = len(self.image_names)

    def __getitem__(self, index):
        _img = Image.open(os.path.join(self.img_root, self.image_names[index])).convert('RGB')
                
        if self.input_transform is not None:
            _img = self.input_transform(_img)
                
        return _img, self.image_names[index]

    def __len__(self):
        return self.data_num
