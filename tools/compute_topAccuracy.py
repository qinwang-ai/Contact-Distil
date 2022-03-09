from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import pandas as pd
import argparse
import os
import numpy as np


class ContactPredictionMetrics(Metric):
    def __init__(self, range_list=["Medium", "Long", "Medium-Long"], k_list=[1, 0.5, 0.2, 0.1], output_transform=lambda x: x, device="cpu"):
        """
        :param range_list: the dist of two residues in their sequence: "Short", "Medium", "Long", "All"
        :param k_list: top ratio - L/k
        :param output_transform:
        :param device:
        """
        self.num_samples = 0
        self.range_list = range_list
        self.k_list = k_list
        self.records = {}
        self.counts = {}
        self.metrics = {}
        self.running_average_metrics = {}
        for r in self.range_list:
            self.counts[r] = {}
            self.metrics[r] = {}
            self.running_average_metrics[r] = {}
            for k in self.k_list:
                self.counts[r][k] = {"TP": [], "TN": [], "FP": [], "FN": []}
                self.metrics[r][k] = {"Precision": -1, "Accuracy": -1}
                # x at 2021.8.5 save running average value.
                #self.running_average_metrics[r][k] = {"Precision": 0, "Accuracy": 0}
                #self.alpha = 0.98

        super(ContactPredictionMetrics, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.num_samples = 0
        self.records = {}
        self.counts = {}
        #self.metrics = {}
        for r in self.range_list:
            self.counts[r] = {}
            #self.metrics[r] = {}
            for k in self.k_list:
                # x at 2021.8.5 save running average value. 一些无法计算的用running average的值替代。仅仅为了running average
                """
                self.running_average_metrics[r][k]["Precision"] = self.running_average_metrics[r][k]["Precision"] * self.alpha + self.metrics[r][k]["Precision"] * (1 - self.alpha)
                self.running_average_metrics[r][k]["Accuracy"] = self.running_average_metrics[r][k]["Accuracy"] * self.alpha + self.metrics[r][k]["Accuracy"] * (1 - self.alpha)
                self.metrics[r][k] = {"Precision": self.running_average_metrics[r][k]["Precision"], "Accuracy": self.running_average_metrics[r][k]["Accuracy"]}
                #"""

                self.counts[r][k] = {"TP": [], "TN": [], "FP": [], "FN": []}
                #self.metrics[r][k] = {"Precision": 0, "Accuracy": 0}   # for some sample which has no whole range


    @reinit__is_reduced
    def update(self, output):
        """
        :param pd_contacts: scores 0~1
        :param gt_contacts: 0s or 1s, otherwise you should add gt_contact = gt_contact.gt(0) & gt_contact.lt(8)
        :return:
        """
        pd_contacts, gt_contacts = output
        pd_contacts = pd_contacts.detach().cpu()

        # for multi-class logits
        if len(pd_contacts.shape) == 4:
            pd_contacts = torch.softmax(pd_contacts, dim=-1)[:, :, :, -1]

        gt_contacts = [gt_contact.detach().cpu() for gt_contact in gt_contacts]
        for index, gt_contact in enumerate(gt_contacts):
            #gt_contact = gt_contact.gt(0) & gt_contact.lt(8)
            pd_contact = pd_contacts[index][0:gt_contact.shape[0], 0:gt_contact.shape[1]]

            gt_contact_non_negative_mask = gt_contact.ge(0)

            length = gt_contact.shape[0]
            for r in self.range_list:
                # create different mask for different range and mask the data
                mask = self.generate_mask(r, length)

                # x at 2021.5.30 consider negative label
                mask = mask * gt_contact_non_negative_mask
                num_valid = torch.sum(mask).int().item()
                if num_valid == 0:
                    continue

                masked_pd_contact = pd_contact * mask
                masked_gt_contact = gt_contact * mask

                # flatten 2d contact map to 1d vector
                pd_contact_flatten = masked_pd_contact.flatten()
                gt_contact_flatten = masked_gt_contact.flatten()

                # sort prediction
                sort_value, sort_index = torch.sort(pd_contact_flatten, descending=True)

                for k in self.k_list:
                    k_value = min(int(k * length), num_valid)
                    k_sort_index = sort_index[0: k_value]
                    pick_pd_contact_flatten = pd_contact_flatten.index_select(0, k_sort_index).gt(0.5)
                    pick_gt_contact_flatten = gt_contact_flatten.index_select(0, k_sort_index).eq(1)

                    tp = (pick_pd_contact_flatten & pick_gt_contact_flatten).sum().unsqueeze(0)
                    tn = (~pick_pd_contact_flatten & ~pick_gt_contact_flatten).sum().unsqueeze(0)
                    fp = (pick_pd_contact_flatten & ~pick_gt_contact_flatten).sum().unsqueeze(0)
                    fn = (~pick_pd_contact_flatten & pick_gt_contact_flatten).sum().unsqueeze(0)

                    # if use these two, the precision will be the referenced metric
                    # I can not understand why we use top L as denominator
                    tp = pick_gt_contact_flatten.sum().unsqueeze(0)
                    fp = k_value - tp   # to make tp + fp = k_value

                    self.counts[r][k]["TP"].append(tp)
                    self.counts[r][k]["TN"].append(tn)
                    self.counts[r][k]["FP"].append(fp)
                    self.counts[r][k]["FN"].append(fn)

                    self.counts[r][k]["TP"] = [torch.cat(self.counts[r][k]["TP"], dim=0)]
                    self.counts[r][k]["TN"] = [torch.cat(self.counts[r][k]["TN"], dim=0)]
                    self.counts[r][k]["FP"] = [torch.cat(self.counts[r][k]["FP"], dim=0)]
                    self.counts[r][k]["FN"] = [torch.cat(self.counts[r][k]["FN"], dim=0)]


    def compute(self):
        r_map = {"All": 0, "Short": 1, "Medium": 2, "Long": 3, "Medium-Long": 4}  # map str to int
        output = []
        for r in self.range_list:
            for k in self.k_list:
                if len(self.counts[r][k]["TP"]) != 0:   # for some sample which has no whole range
                    tp = self.counts[r][k]["TP"][0].float()  # .sum()
                    tn = self.counts[r][k]["TN"][0].float()  # .sum()
                    fp = self.counts[r][k]["FP"][0].float()  # .sum()
                    fn = self.counts[r][k]["FN"][0].float()  # .sum()
                    self.metrics[r][k]["Precision"] = (tp / (tp + fp).clamp(min=1E-12)).mean()
                    self.metrics[r][k]["Accuracy"] = ((tp + tn) / (tp + tn + fp + fn).clamp(min=1E-12)).mean()

                output.append([r_map[r], k, self.metrics[r][k]["Precision"]])
        output = torch.Tensor(output)

        return output

        # return {"precision": self.precision, "accuracy": self.accuracy}   for RunningAverage you must return a

    def generate_mask(self, range_type, length):
        """
        :param range_type: all, short, medium, long
        :param length:
        :return:
        """
        sub_masks = []
        if "All" in range_type:
            sub_mask = torch.ones((length, length))
            sub_mask = torch.triu(sub_mask, 0)
            sub_masks.append(sub_mask)

        if "Short" in range_type:
            sub_mask = torch.ones((length, length))
            sub_mask = torch.triu(sub_mask, 6) - torch.triu(sub_mask, 12)
            sub_masks.append(sub_mask)

        if "Medium" in range_type:
            sub_mask = torch.ones((length, length))
            sub_mask = torch.triu(sub_mask, 12) - torch.triu(sub_mask, 24)
            sub_masks.append(sub_mask)

        if "Long" in range_type:
            sub_mask = torch.ones((length, length))
            sub_mask = torch.triu(sub_mask, 24)
            sub_masks.append(sub_mask)

        mask = 0
        for sub_mask in sub_masks:
            mask = mask + sub_mask

        return mask

    def comeputeTopAccuracy(self, gt_contact, pd_contact):
        self.reset()
        self.update((gt_contact.unsqueeze(0), [pd_contact]))
        output = self.compute()
        self.reset()
        return output



parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--pd_path', type=str, default="none",
                    help='file path to predictions, either contact prob or argmax map')

parser.add_argument('--gt_path',type=str, default="none",
                    help='file path to ground truth')

parser.add_argument('--pd_type',type=str, default="2d",
                    help='file path to ground truth')

parser.add_argument('--pd_dir', type=str, default="none",
                    help='dir of predictions, either contact prob or argmax map')

parser.add_argument('--gt_dir',type=str, default="none",
                    help='dir of ground truth')

"""
example:
1.For file
python compute_topAccuracy.py --pd_path=PATH_to_PD --gt_path=PATH_to_GT

2.For folder
python compute_topAccuracy.py --pd_dir=PD_Folder --gt_dir=GT_Folder

"""



def read_2d_map(ann2d_path):
    information_list = []
    with open(ann2d_path, "r") as f:
        for line in f:
            information = []
            raw_information = line.strip("\n").split(" ")
            for sub_info in raw_information:
                if sub_info != "":
                    information.append(sub_info)
            information_list.append(information)

    cm_shape = (int(information[0]) + 1, int(information[1]) + 1)
    information_numpy = np.array(information_list)
    distance_map = information_numpy[:, 2].reshape(cm_shape).astype(np.float)

    contact_map = distance_map * (distance_map < 0) + (distance_map < 8) * (distance_map >= 0)
    contact_map = contact_map.astype(np.int)

    return contact_map

if __name__=="__main__":
    args = parser.parse_args()
    if args.pd_path != "none" and args.gt_path != "none":
        CPM = ContactPredictionMetrics()

        if args.pd_type == "npy":
            pd_contact = np.load(args.pd_path)
        else:
            pd_contact = read_2d_map(args.pd_path)

        gt_contact = read_2d_map(args.gt_path)

        pd_contact = torch.Tensor(pd_contact)
        gt_contact = torch.Tensor(gt_contact)

        #CPM.update((gt_contact.unsqueeze(0), [pd_contact]))
        CPM.update((pd_contact.unsqueeze(0), [gt_contact]))
        metric = CPM.compute()

        cp_precision = {}
        # cp_accuracy = {}
        r_map = {0: "A", 1: "S", 2: "M", 3: "L", 4: "ML"}
        for i in range(metric.shape[0]):
            r_key = metric[i][0].item()
            k_key = metric[i][1].item()
            key = "{}-{:.1f}L".format(r_map[int(r_key)], k_key)
            cp_precision[key] = metric[i][2].item()
        print(cp_precision)

    if args.pd_dir != "none" and args.gt_dir != "none":
        CPM = ContactPredictionMetrics()

        file_list = os.listdir(args.pd_dir)
        for filename in file_list:
            pd_path = os.path.join(args.pd_dir, filename)
            gt_path = os.path.join(args.gt_dir, filename)

            if args.pd_type == "npy":
                pd_contact = np.load(pd_path)
            else:
                pd_contact = read_2d_map(pd_path)

            gt_contact = read_2d_map(gt_path)

            pd_contact = torch.Tensor(pd_contact)
            gt_contact = torch.Tensor(gt_contact)

            CPM.update((pd_contact.unsqueeze(0), [gt_contact]))

        metric = CPM.compute()

        cp_precision = {}
        r_map = {0: "A", 1: "S", 2: "M", 3: "L", 4: "ML"}
        for i in range(metric.shape[0]):
            r_key = metric[i][0].item()
            k_key = metric[i][1].item()
            key = "{}-{:.1f}L".format(r_map[int(r_key)], k_key)
            cp_precision[key] = metric[i][2].item()
        print(cp_precision)




