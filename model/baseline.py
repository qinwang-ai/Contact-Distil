# encoding: utf-8

import torch

from .backbones import choose_backbone
from .contact_predictor import choose_contact_predictor
from .weights_init import *
from ptflops import get_model_complexity_info

class Baseline(nn.Module):
    def __init__(self,
                 backbone_name,
                 contact_predictor_name="none",
                 backbone_frozen=False
                 ):
        super(Baseline, self).__init__()
        # 0.Configuration
        self.backbone_name = backbone_name
        self.contact_predictor_name = contact_predictor_name
        self.backbone_frozen = backbone_frozen

        # 1.Build backbone
        self.backbone, self.backbone_alphabet = choose_backbone(self.backbone_name)
        self.backbone.lm_head.requires_grad_(False)

        self.control_parameter_dict = {"return_contacts": False, "need_head_weights": False, "repr_layers": []}

        # 2.Build Contact Predictor
        self.ct_predictor, new_cp_dict = choose_contact_predictor(
            self.contact_predictor_name, self.backbone.args, self.backbone_alphabet
        )
        self.control_parameter_dict.update(new_cp_dict)

        # Initialization of parameters
        #self.backbone.apply(weights_init_kaiming)
        #self.backbone.apply(weights_init_classifier) # maybe with classifier itself
        if self.ct_predictor is not None:
            self.ct_predictor.apply(weights_init_classifier)


    def forward(self, x, profiles):
        need_head_weights = self.control_parameter_dict["need_head_weights"]
        repr_layers = self.control_parameter_dict["repr_layers"]
        return_contacts = self.control_parameter_dict["return_contacts"]

        if self.backbone_frozen == True:
            self.backbone.eval()
            with torch.no_grad():
                results = self.backbone(x, need_head_weights=need_head_weights, repr_layers=repr_layers, return_contacts=return_contacts)
        else:
            results = self.backbone(x, need_head_weights=need_head_weights, repr_layers=repr_layers, return_contacts=return_contacts)

        if self.ct_predictor is not None:
            if self.contact_predictor_name == "LR" or "MultiLayerTiedRowAttention" in self.contact_predictor_name:
                attentions = results["attentions"] if results.get("attention") is not None else results["row_attentions"]
                results["contacts"] = self.ct_predictor(x, attentions)
            else:
                if profiles is None:
                    results["contacts"] = self.ct_predictor(x, results['representations'][12])
                else:
                    output = self.ct_predictor(x, results['representations'][12], profiles)
                    if isinstance(output, tuple):
                        results["contacts"] = output[0]
                        results["profile_contacts"] = output[1]
                    else:
                        results["contacts"] = output

        # postprocess - eliminate cls_idx
        """
        if self.backbone.prepend_bos == True:
            for key in results.keys():
                if key == "sses":
                    results[key] = results[key][:, 1:results[key].shape[1], :]
                elif key == "contacts":
                    results[key] = results[key][:, 1:results[key].shape[1], 1:results[key].shape[2]]
        """

        return results

    # load parameter
    def load_param(self, load_choice, model_path):
        param_dict = torch.load(model_path, map_location='cpu')  #["model"]
        if param_dict.get("model") is not None and param_dict.get("args") is not None:
            print("Does not reload weights from official pre-trained file!")
            return 1

        if load_choice == "Backbone":
            base_dict = self.backbone.state_dict()
            for i in param_dict:
                module_name = i.replace("backbone.", "")
                if module_name not in self.backbone.state_dict():
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue
                self.backbone.state_dict()[module_name].copy_(param_dict[i])
            print("Complete Load Weight")

        elif load_choice == "Overall":
            overall_dict = self.state_dict()
            for i in param_dict:
                if i in self.state_dict():
                    try:
                        self.state_dict()[i].copy_(param_dict[i])
                    except:
                        print("connot load {}".format(i))
                elif "base."+i in self.state_dict():
                    self.state_dict()["base."+i].copy_(param_dict[i])
                elif "backbone."+i in self.state_dict():
                    self.state_dict()["backbone."+i].copy_(param_dict[i])
                elif i.replace("base", "backbone") in self.state_dict():
                    self.state_dict()[i.replace("base", "backbone")].copy_(param_dict[i])
                else:
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue
            print("Complete Load Weight")

        elif load_choice == "None":
            print("Do not reload Weight by myself.")


    def count_param(model, input_shape=(3, 224, 224)):
        with torch.cuda.device(0):
            flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            return ('{:<30}  {:<8}'.format('Computational complexity: ', flops)) + (
                '{:<30}  {:<8}'.format('Number of parameters: ', params))