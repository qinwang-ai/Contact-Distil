# encoding: utf-8
import logging

import os

import torch
import torch.nn as nn
from ignite.engine import Engine, Events

from ignite.metrics import Accuracy
from ignite.metrics import Precision
from ignite.metrics import Recall
from ignite.metrics import ConfusionMatrix
from ignite.metrics import MeanSquaredError

from sklearn.metrics import roc_curve, auc

import numpy as np

from metric import *

global save_dir

def create_supervised_evaluator(model, metrics, loss_fn=None, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if next(model.parameters()).is_cuda:
            pass
        else:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # fetch data
            data, anns = batch  # corresponding to collate_fn
            tokens = data["tokens"]
            gt_contacts = anns["contact_maps"] if anns is not None else None
            profiles = anns["profiles"] if (anns is not None and anns.get("profiles")) is not None else None
            pseudo_profiles = anns["pseudo_profiles"] if (anns is not None and anns.get(
                "pseudo_profiles")) is not None else None

            # place data in CUDA
            tokens = tokens.to(device) if torch.cuda.device_count() >= 1 and tokens is not None else tokens
            if torch.cuda.device_count() >= 1 and gt_contacts is not None:
                if isinstance(gt_contacts, list):
                    gt_contacts = [gt_contact.to(device) for gt_contact in gt_contacts]
                    if profiles is not None:
                        profiles = torch.nn.utils.rnn.pad_sequence(profiles, batch_first=True, padding_value=0).to(
                            device)
                    if pseudo_profiles is not None:
                        pseudo_profiles = torch.nn.utils.rnn.pad_sequence(pseudo_profiles, batch_first=True, padding_value=0)[:, 0:-1, :].to(device)
                else:
                    gt_contacts.to(device)
                    if profiles is not None:
                        profiles.to(device)
                    if pseudo_profiles is not None:
                        pseudo_profiles.to(device)

            # forward propagation
            results = model(tokens, pseudo_profiles)
            pd_contacts = results["contacts"] if results.get("contacts") is not None else None


            # save metric csv
            save_single_metric = 1
            if save_single_metric == True:
                from tools.compute_topAccuracy import ContactPredictionMetrics
                CPM = ContactPredictionMetrics()
                CPM.update((pd_contacts, gt_contacts))
                metric = CPM.compute()
                cp_precision = {}
                r_map = {0: "A", 1: "S", 2: "M", 3: "L", 4: "ML"}
                for i in range(metric.shape[0]):
                    r_key = metric[i][0].item()
                    k_key = metric[i][1].item()
                    key = "{}-{:.1f}L".format(r_map[int(r_key)], k_key)
                    cp_precision[key] = metric[i][2].item()
                # print(cp_precision)

                global save_dir

                f_filename = anns["filenames"][0]
                f_depth = tokens.shape[1]

                if engine.state.iteration == 1:
                    with open(save_dir, "w") as f:
                        f.write("{} {:.3f} {:.3f} {:.3f}\n".format(f_filename, cp_precision["L-1.0L"],
                                                                   cp_precision["L-0.5L"],
                                                                   cp_precision["L-0.2L"]))
                else:
                    with open(save_dir, "a") as f:
                        f.write("{} {:.3f} {:.3f} {:.3f}\n".format(f_filename, cp_precision["L-1.0L"],
                                                                   cp_precision["L-0.5L"],
                                                                   cp_precision["L-0.2L"]))

            # save npy
            save_single_npy = 0
            if save_single_npy == True:
                prediction = torch.argmax(pd_contacts.detach(), dim=-1)[0].cpu() * gt_contacts[0].ge(0).cpu()
                prediction = torch.triu(prediction, 1).T + torch.triu(prediction)
                prediction = prediction.numpy()
                save_folder, _ = os.path.split(save_dir)
                save_numpy_path = os.path.join(save_folder, anns["filenames"][0] + ".npy")
                np.save(save_numpy_path, prediction)


            return {"pd_contacts": pd_contacts, "gt_contacts": gt_contacts}

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_inference(
        cfg,
        model,
        test_loader,
        loss_fn=None,
        target_set_name="test",
        plot_flag=False
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("classification.inference")
    logging._warn_preinit_stderr = 0
    logger.info("Enter inferencing for {} set".format(target_set_name))

    metrics_eval = {}

    global save_dir
    save_dir = os.path.join(cfg.SOLVER.OUTPUT_DIR, cfg.DATA.DATASETS.NAMES + ".txt")

    # add seperate metrics
    lossKeys = cfg.LOSS.TYPE.split(" ")
    if "counts_regression_loss" in lossKeys:
        lossKeys.append("counts_classification_loss")

    for lossName in lossKeys:
        if lossName == "contact_prediction_loss" or lossName == "contact_distillation_loss" or lossName == "profile_contact_prediction_loss":
            metrics_eval["contact_prediction_metrics"] = ContactPredictionMetrics(output_transform=lambda x: (x["pd_contacts"], x["gt_contacts"]))
        else:
            raise Exception('expected METRIC_LOSS_TYPE should not be {}'.format(cfg.LOSS.TYPE))

    evaluator = create_supervised_evaluator(model, metrics=metrics_eval, loss_fn=loss_fn, device=device)

    Eval_Record = {}

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if engine.state.iteration % 10 == 0:
            print("Iteration[{}/{}]".format(engine.state.iteration, len(test_loader)))


    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_inference_results(engine):
        info = "Test Results\n"

        # 1. Temporary Record Dict (Can be used for Tensorboard Summary)
        if engine.state.metrics.get("contact_prediction_metrics") != None:
            # format contact prediction metrics
            Eval_Record["Contact Prediction"] = {"Precision": {}}

            cp_precision = {}
            # cp_accuracy = {}
            r_map = {0: "A", 1: "S", 2: "M", 3: "L", 4: "ML"}
            for i in range(engine.state.metrics['contact_prediction_metrics'].shape[0]):
                r_key = engine.state.metrics['contact_prediction_metrics'][i][0].item()
                k_key = engine.state.metrics['contact_prediction_metrics'][i][1].item()
                key = "{}-{:.1f}L".format(r_map[int(r_key)], k_key)
                cp_precision[key] = engine.state.metrics['contact_prediction_metrics'][i][2].item()
                # cp_accuracy[key] = engine.state.metrics['contact_prediction_metrics'][i][3].item()

            Eval_Record["Contact Prediction"]["Precision"] = cp_precision

            cp_precision = {}
            for key in Eval_Record["Contact Prediction"]["Precision"].keys():
                cp_precision[key] = "{:.3f}".format(Eval_Record["Contact Prediction"]["Precision"][key])
            info += ">> Contact Prediction - Precision: {}\n".format(cp_precision)

        logger.info(info.replace("'", "").strip("\n"))

    evaluator.run(test_loader)

    return Eval_Record



