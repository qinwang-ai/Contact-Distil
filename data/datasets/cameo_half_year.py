# encoding: utf-8

import os
import pandas as pd
from torch.utils.data import Dataset

from .utils import *

class CAMEO_HALF_YEAR(Dataset):
    def __init__(self, root, data_type="seq", set_name="train", transform=None, max_length_limit=512, depth=64, profile_type="none"):
        """
        :param root: root path of dataset - CATH. however not all of stuffs under this root path
        :param data_type: seq, msa
        :param label_type: 1d, 2d? To Do
        :param set_name: "train", "valid", "test"
        :param transform: useless so far
        """
        self.root = root
        self.data_type = data_type
        self.set_name = set_name
        self.transform = transform
        self.max_length_limit = max_length_limit
        self.seq_ext_type = ".fasta"
        # self.depth = depth
        # x at 2021.9.6 for teahcer-student
        if depth == 0:
            self.depth = 60 if self.set_name == "train" else 10
        else:
            self.depth = depth

        self.seq_dir = os.path.join(self.root, "seq")
        if self.set_name == "train":
            self.msa_dir = os.path.join(self.root,
                                        "msa_downsample")  # "/share/zhengliangzhen/datasets/AAAI_202108/msa_jackhmm_uniref90_201907_e0.01_iter3"
        else:
            self.msa_dir = os.path.join(self.root, "msa_downsample_{}".format(self.depth))
        self.ann1d_dir = os.path.join(self.root, "dump_1d")
        self.ann2d_dir = os.path.join(self.root, "dump_2d")
        if profile_type == "none":
            self.profile_dir = None
            self.pseudo_profile_dir = None
        elif profile_type == "profile_gt":
            self.profile_dir = os.path.join(self.root, profile_type)
            self.pseudo_profile_dir = None
        elif profile_type == "profile_pd":
            self.profile_dir = None
            self.pseudo_profile_dir = os.path.join(self.root, profile_type)
        elif profile_type == "both":
            self.profile_dir = os.path.join(self.root, "profile_gt")
            self.pseudo_profile_dir = os.path.join(self.root, "profile_pd")
        self.name_path = os.path.join(self.root, "annotation", "annotation.csv")

        self.batch_converter = None
        self.data_reader = DataReader(self.data_type, msa_nseq=self.depth, use_cache=1, cache_dir=os.path.join(self.root, "cache"))

        self.sse_classes = self.data_reader.sse_classes

        self.seqs, self.msas, self.anns, self.stats = self.__dataset_info(self.name_path, self.seq_dir, self.msa_dir,
                                                                          self.ann1d_dir, self.ann2d_dir,
                                                                          self.profile_dir, self.pseudo_profile_dir)

    def __getitem__(self, index):
        data = self.data_reader.read_data(self.msas[index])
        anns = self.data_reader.read_anns(self.anns[index])
        return data, anns, self.batch_converter

    def __len__(self):
        return len(self.seqs)

    def __dataset_info(self, annotation_path, seq_dir, msa_dir, ann1d_dir, ann2d_dir, profile_dir=None, pseudo_profile_dir=None):
        """
        :param name_path: txt record name list for specific set_name
        :param seq_dir:
        :param msa_dir:
        :param ann1d_dir:
        :param ann2d_dir:
        :return:
        """
        seqs = []
        msas = []
        anns = []

        selected_indices = []

        src_df = pd.read_csv(annotation_path)
        for index, row in src_df.iterrows():
            name = row["filename"]
            extension = row["extension"]
            length = row["length"]
            depth = row["depth"]

            #if name != "2021-03-07_00000206_1" and name != "2021-03-27_00000167_2":
            #if name != "2021-04-03_00000200_1" and name != "2021-04-17_00000111_1":
            #    continue

            seq_path = os.path.join(seq_dir, name + self.seq_ext_type)
            msa_path = os.path.join(msa_dir, name + ".a3m")
            ann1d_path = os.path.join(ann1d_dir, name)
            ann2d_path = os.path.join(ann2d_dir, name + ".2d")

            if os.path.exists(seq_path) != True:
                print("{} doesn't exist.".format("SEQ: " + seq_path))
                continue

            if os.path.exists(msa_path) != True:
                print("{} doesn't exist.".format("MSA: " + msa_path))
                continue

            #if os.path.exists(ann2d_path) != True:
            #    print("{} doesn't exist.".format("2DL: " + ann2d_path))
            #    continue

            if int(length) > self.max_length_limit and self.max_length_limit != -1:
                continue

            seqs.append(seq_path)
            msas.append(msa_path)
            # anns.append({"1d": ann1d_path, "2d": ann2d_path})

            if profile_dir is None and pseudo_profile_dir is None:
                anns.append({"2d": ann2d_path})
            elif profile_dir is not None:
                profile_path = os.path.join(profile_dir, name + ".npy")
                anns.append({"2d": ann2d_path, "profile": profile_path})
            elif pseudo_profile_dir is not None:
                pseudo_profile_path = os.path.join(pseudo_profile_dir, name + ".npy")
                anns.append({"2d": ann2d_path, "pseudo_profile": pseudo_profile_path})
            else:
                pass

            selected_indices.append(index)

            # truncation
            # if len(statistics) > 10:
            #    break

        print("{} Dataset Info:".format(self.set_name))
        print("Length-Frequency Table")
        selected_df = src_df.iloc[selected_indices]
        print(selected_df["length"].describe())  # value_counts())
        print("Depth-Frequency Table")
        selected_df = src_df.iloc[selected_indices]
        print(selected_df["depth"].describe())

        return seqs, msas, anns, src_df

    def get_batch_converter(self, alphabet):
        self.alphabet = alphabet
        self.batch_converter = BatchConverter(alphabet, self.data_type)

    # reset max_length
    def reset_max_length_limit(self, max_length_limit):
        self.max_length_limit = max_length_limit
        self.seqs, self.msas, self.anns, self.stats = self.__dataset_info(self.name_path, self.seq_dir, self.msa_dir,
                                                                          self.ann1d_dir, self.ann2d_dir)