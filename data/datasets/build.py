from .cameo_half_year import CAMEO_HALF_YEAR

def build_dataset(dataset_name, set_name, root_path, transforms=None):
    """
    :param dataset_name: the name of dataset
    :param root_path: data is usually located under the root path
    :param set_name: "train", "valid", "test"
    :param transforms:
    :return:
    """
    if "cameo_half_year" in dataset_name:
        _, data_type, max_length, depth, profile_type = dataset_name.split("-")
        max_length = int(max_length)
        depth = int(depth)
        dataset = CAMEO_HALF_YEAR(root=root_path, data_type=data_type,
                        transform=transforms, max_length_limit=max_length, depth=depth, profile_type=profile_type)
    else:
        raise Exception("Can not build unknown image dataset: {}".format(dataset_name))

    return dataset