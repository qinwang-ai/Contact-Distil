# encoding: utf-8
from .resnet_2d import ResNet2d
from .resnet_with_profilenet import ResnetWithProfileNet

def choose_contact_predictor(contact_predictor_name, backbone_args, backbone_alphabet):
    """
    :param contact_predictor_name:
    :param backbone_args:
    :param backbone_alphabet:
    :return:
    """
    if 'ResNet2d' in contact_predictor_name:
        _, depth_reduction = contact_predictor_name.split("-")
        contact_predictor = ResNet2d(backbone_args, backbone_alphabet, num_classes=2, depth_reduction=depth_reduction)
        backbone_control_parameter_dict = {"return_contacts": False, "need_head_weights": False, "repr_layers": [12]}
    elif 'ResNetWithProfileNet' in contact_predictor_name:
        _, depth_reduction = contact_predictor_name.split("-")
        contact_predictor = ResnetWithProfileNet(backbone_args, backbone_alphabet, num_classes=2, depth_reduction=depth_reduction)
        backbone_control_parameter_dict = {"return_contacts": False, "need_head_weights": False, "repr_layers": [12]}
    elif 'ResNetWithProfile' in contact_predictor_name:
        _, depth_reduction = contact_predictor_name.split("-")
        contact_predictor = ResnetWithProfileNet(backbone_args, backbone_alphabet, num_classes=2, depth_reduction=depth_reduction, with_profilenet=False)
        backbone_control_parameter_dict = {"return_contacts": False, "need_head_weights": False, "repr_layers": [12]}
    elif contact_predictor_name == "none":
        contact_predictor = None
        backbone_control_parameter_dict = {}
        print("Without Independent Contact Predictor!")
    else:
        raise Exception("Wrong Backbone Type!")

    return contact_predictor, backbone_control_parameter_dict