# encoding: utf-8


import esm as esm

def choose_backbone(backbone_name):
    # 1.ESM1b
    if backbone_name == 'esm1b':
        backbone, backbone_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

    # 2.MSA Transformer
    elif backbone_name == 'msa_transformer':
        backbone, backbone_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()

    else:
        raise Exception("Wrong Backbone Type!")

    return backbone, backbone_alphabet