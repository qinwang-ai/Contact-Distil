# encoding: utf-8

from .baseline import Baseline

def build_model(cfg):
    model = Baseline(
        backbone_name=cfg.MODEL.BACKBONE_NAME,
        contact_predictor_name=cfg.MODEL.CONTACT_PREDICTOR_NAME,
        backbone_frozen=cfg.MODEL.BACKBONE_FROZEN,
    )
    return model
