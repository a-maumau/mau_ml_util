from ..templates.data_loader_template import Template_ClassificationDatasetLoader

class ClassificationDatasetLoader(Template_ClassificationDatasetLoader):
    def __init__(self, img_root, img_list_path, dataset_pickle_path=None,
                       input_transform=None,
                       load_all_in_ram=True, img_ext=".jpg", img_convert_type="RGB",
                       pickle_img_key="image", pickle_label_key="label",
                       pickle_path_data=False, pickle_path_relative=False):

        super(ClassificationDatasetLoader, self).__init__(img_root, img_list_path, dataset_pickle_path,
                                                          input_transform,
                                                          load_all_in_ram, img_ext, img_convert_type,
                                                          pickle_img_key, pickle_label_key,
                                                          pickle_path_data, pickle_path_relative)
