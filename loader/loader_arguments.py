"""
    argments which required in data loaders.
"""

def add_loader_arguments(parser):
    """
        parser: argparse.ArgumentParser
    """

    # paths
    parser.add_argument('--img_root', type=str, default="log", help='dir of input image exist.')
    parser.add_argument('--mask_root', type=str, default="log", help='dir of mask image exist.')
    parser.add_argument('--img_list_path', type=str, default=None,
                        help="path of dataset's image set list file that contains a image name for traing and might with labels.")
    parser.add_argument('--dataset_pickle_path', type=str, default=None, help='path of preprocessed dataset which is pickled.')
