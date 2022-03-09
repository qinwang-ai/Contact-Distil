# encoding: utf-8

import argparse
import os
import sys

import torch
from torch.backends import cudnn
import numpy as np
import random

sys.path.append('.')

from data import make_data_loader

from model import build_model
from engine.evaluator import do_inference

from config import cfg
from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter

from utils.tensorboard_logger import record_dict_into_tensorboard


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def eval(cfg, target_set_name="test"):
    # prepare dataset
    train_loader, val_loader, test_loader = make_data_loader(cfg, is_train=False)

    # build model and load parameter
    model = build_model(cfg)
    if os.path.exists(cfg.TEST.WEIGHT) != True:
        if os.path.exists("./pretrained/") != True:
            os.makedirs("./pretrained")
        os.system("wget -O  './pretrained/msa_transformer_model.pth' https://tmp-titan.vx-cdn.com:616/file/613ca738783a8/msa_transformer_model.pth")
    model.load_param("Overall", cfg.TEST.WEIGHT)  #"Overall", "None"

    # pass alphabet to construct batch converter for dataset
    train_loader.dataset.get_batch_converter(model.backbone_alphabet)
    val_loader.dataset.get_batch_converter(model.backbone_alphabet)
    test_loader.dataset.get_batch_converter(model.backbone_alphabet)

    # input data_loader
    if target_set_name == "train":
        input_data_loader = train_loader
    elif target_set_name == "valid":
        input_data_loader = val_loader
    elif target_set_name == "test":
        input_data_loader = test_loader
    else:
        raise Exception("Wrong Dataset Name!")

    # build and launch engine for evaluation
    Eval_Record = do_inference(cfg,
                               model,
                               input_data_loader,
                               None,
                               target_set_name=target_set_name,
                               plot_flag=True)

    # logging with tensorboard summaryWriter
    model_epoch = cfg.TEST.WEIGHT.split('/')[-1].split('.')[0].split('_')[-1]
    model_iteration = len(train_loader) * int(model_epoch) if model_epoch.isdigit() == True else 0

    writer_test = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/eval_" + target_set_name)
    record_dict_into_tensorboard(writer_test, Eval_Record, model_iteration)
    writer_test.close()

    # record in csv
    csv_name = "metrics"
    import pandas as pd
    if Eval_Record.get("Contact Prediction") is not None:
        sheet_name = "Contact-Prediction"
        col_names = ["dataset"]
        value = [cfg.DATA.DATASETS.NAMES]
        for k in Eval_Record["Contact Prediction"]["Precision"].keys():
            col_names.append(k)
            value.append(Eval_Record["Contact Prediction"]["Precision"][k])
        df = pd.DataFrame([value], columns=col_names)

        xls_filename = os.path.join(cfg.SOLVER.OUTPUT_DIR, "{}.xlsx".format(csv_name))
        if os.path.exists(xls_filename) != True:
            with pd.ExcelWriter(xls_filename, engine="openpyxl", mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(xls_filename, engine="openpyxl", mode='a') as writer:
                wb = writer.book
                if sheet_name in wb.sheetnames:
                    old_df = pd.read_excel(xls_filename, sheet_name=sheet_name, index_col=0)
                    # remove old sheet, otherwise generate new sheets with suffix "1", "2",...
                    wb.remove(wb[sheet_name])
                    df = pd.concat([old_df, df], axis=0, ignore_index=True)
                    df.to_excel(writer, sheet_name=sheet_name)
                else:
                    df.to_excel(writer, sheet_name=sheet_name)

    if Eval_Record.get("Secondary Structure Prediction") is not None:
        sheet_name = "Secondary-Structure-Prediction"
        col_names = ["dataset"]
        value = [cfg.DATA.DATASETS.NAMES]
        col_names.append("Accuracy")
        value.append(Eval_Record["Secondary Structure Prediction"]["Accuracy"])

        for k in Eval_Record["Secondary Structure Prediction"]["Precision"].keys():
            col_names.append("Precision-" + k)
            value.append(Eval_Record["Secondary Structure Prediction"]["Precision"][k])
        for k in Eval_Record["Secondary Structure Prediction"]["Recall"].keys():
            col_names.append("Recall-" + k)
            value.append(Eval_Record["Secondary Structure Prediction"]["Recall"][k])
        df = pd.DataFrame([value], columns=col_names)

        xls_filename = os.path.join(cfg.SOLVER.OUTPUT_DIR, "{}.xlsx".format(csv_name))
        if os.path.exists(xls_filename) != True:
            with pd.ExcelWriter(xls_filename, engine="openpyxl", mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(xls_filename, engine="openpyxl", mode='a') as writer:
                wb = writer.book
                if sheet_name in wb.sheetnames:
                    old_df = pd.read_excel(xls_filename, sheet_name=sheet_name, index_col=0)
                    # remove old sheet, otherwise generate new sheets with suffix "1", "2",...
                    wb.remove(wb[sheet_name])
                    df = pd.concat([old_df, df], axis=0, ignore_index=True)
                    df.to_excel(writer, sheet_name=sheet_name)
                else:
                    df.to_excel(writer, sheet_name=sheet_name)


def main():
    parser = argparse.ArgumentParser(description="Classification Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--target_set", default="test", help="name of target dataset: train, valid, test, all", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TRAIN.DATALOADER.IMS_PER_BATCH = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.VAL.DATALOADER.IMS_PER_BATCH = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH * cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.TEST.DATALOADER.IMS_PER_BATCH = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH * cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.freeze()

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("classification", output_dir, "eval_on_{}_{}".format(cfg.DATA.DATASETS.NAMES, args.target_set), 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join("%s"%i for i in cfg.MODEL.DEVICE_ID)   # int tuple -> str # cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    logger.info("Eval on the {} dataset".format(args.target_set))
    if args.target_set == "train" or args.target_set == "valid" or args.target_set == "test":
        eval(cfg, args.target_set)
    elif args.target_set == "all":
        eval(cfg, "train")
        eval(cfg, "valid")
        eval(cfg, "test")
    else:
        raise Exception("Wrong dataset name with {}".format(args.dataset_name))


if __name__ == '__main__':
    seed_torch(2018)
    main()
