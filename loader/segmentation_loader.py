from ..templates.data_loader_template import Template_SegmentationDatasetLoader

class SegmentationDatasetLoader(Template_SegmentationDatasetLoader):
    def __init__(self, img_root, mask_root, img_list_path=None, dataset_pickle_path=None,
                       pair_transform=None, input_transform=None, target_transform=None,
                       load_all_in_ram=True, img_ext=".jpg", mask_ext=".png", img_convert_type="RGB",
                       pickle_img_key="image", pickle_mask_key="mask",
                       pickle_path_data=False, pickle_path_relative=False,
                       return_original=False):

        super(SegmentationDatasetLoader, self).__init__(img_root, mask_root, img_list_path, dataset_pickle_path,
                                                        pair_transform, input_transform, target_transform,
                                                        load_all_in_ram, img_ext, mask_ext, img_convert_type,
                                                        pickle_img_key, pickle_mask_key,
                                                        pickle_path_data, pickle_path_relative,
                                                        return_original)
