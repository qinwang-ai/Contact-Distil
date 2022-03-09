from typing import Tuple, List
import string
import itertools
from Bio import SeqIO
import numpy as np
import os
import pickle
import random

class DataReader(object):
    def __init__(self, data_type="seq", msa_nseq=0, sse_type=3, use_cache=0, cache_dir=None):
        """
        :param data_type: "seq", "msa"
        :param msa_nseq: Reads the first nseq sequences from an MSA file if it is in "msa" data type.
        """
        # This is an efficient way to delete lowercase characters and insertion characters from a string
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        self.translation = str.maketrans(deletekeys)
        self.data_type = data_type
        self.msa_nseq = msa_nseq

        # x at 2021.9.1 cache
        self.use_cache = use_cache
        self.cache_dir = cache_dir   #"/ssdcache/abc/trRosetta/cache"
        self.cache = {"data":{}, "anns":{}}

        # sse
        #self.sse_classes = ["L", "T", "E", "S", "H", "G", "I", "B"]
        #self.class_to_label = {}
        #for c in self.sse_classes:
        #    self.class_to_label[c] = self.sse_classes.index(c)
        if sse_type == 3:
            self.sse_classes = ["H", "E", "C"]
            self.class_to_label = {"H": 0, "E": 1, "L": 2, "T": 2, "S": 2, "G": 2, "I": 2, "B": 2}
        elif sse_type == 8:
            self.sse_classes = ["H", "E", "L", "T", "S", "G", "I", "B"]
            self.class_to_label = {"H": 0, "E": 1, "L": 2, "T": 3, "S": 4, "G": 5, "I": 6, "B": 7}
        else:
            raise Exception("Wrong Type SSE class Type")

    def read_data(self, data_path):
        # x use cache
        """
        if self.use_cache == 1 and self.cache["data"].get(data_path) is not None:
            return self.cache["data"][data_path]

        _, filename = os.path.split(data_path)
        cache_path = os.path.join(self.cache_dir, filename.replace(".a3m", "_a3m.pickle"))
        if os.path.exists(cache_path) == True:
            #data = np.load(cache_path)
            with open(cache_path, 'rb') as handle:
                data = pickle.load(handle)
            self.cache["data"][data_path] = data
            return data
        """
        if self.data_type == "seq":
            data = self.read_sequence(data_path)
        elif self.data_type == "msa":
            data = self.read_msa(data_path, self.msa_nseq, mode="random")
        else:
            raise Exception("Wrong Type!")

        # x use cache
        """
        if self.use_cache == 1 and self.cache["data"].get(data_path) is None:
            self.cache["data"][data_path] = data
            #np.save(cache_path, data)
            with open(cache_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("save")
        """

        return data

    def read_anns(self, ann_paths):
        # x use cache
        if self.use_cache == 1 and self.cache["anns"].get(ann_paths["2d"]) is not None:
            return self.cache["anns"][ann_paths["2d"]]
        _, filename = os.path.split(ann_paths["2d"])
        cache_path = os.path.join(self.cache_dir, filename.replace(".2d", "_2d.pickle"))
        if self.use_cache == 1 and os.path.exists(cache_path) == True:
            #annotations = np.load(cache_path)
            with open(cache_path, 'rb') as handle:
                annotations = pickle.load(handle)

            if ann_paths.get("profile") is not None:
                annotations["profile"] = np.load(ann_paths["profile"])

            if ann_paths.get("pseudo_profile") is not None:
                annotations["pseudo_profile"] = np.load(ann_paths["pseudo_profile"])

            # filename
            _, filename = os.path.split(ann_paths["2d"])
            filename_pre, ext = os.path.splitext(filename)
            annotations["filename"] = filename_pre

            self.cache["anns"][ann_paths["2d"]] = annotations

            return annotations

        annotations = {}
        if ann_paths.get("1d") is not None:
            ann1d_path = ann_paths["1d"]
            sse = self.read_sse(ann1d_path)
            annotations["sse"] = sse

            # filename
            _, filename = os.path.split(ann1d_path)
            filename_pre, ext = os.path.splitext(filename)
            annotations["filename"] = filename_pre

        if ann_paths.get("2d") is not None:
            ann2d_path = ann_paths["2d"]
            distance_map, omega_map, theta_map, phi_map = self.read_2d_map(ann2d_path)
            annotations["distance_map"] = distance_map

            contact_map = distance_map * (distance_map < 0) + (distance_map < 8) * (distance_map >= 0)
            contact_map = contact_map.astype(np.int8)
            annotations["contact_map"] = contact_map

            # discretization
            dist_bin_map = self.discretize_dist(distance_map)
            annotations["dist_bin_map"] = dist_bin_map
            no_contact_mask = dist_bin_map == 0
            omega_bin_map = self.discretize_omega(omega_map, no_contact_mask)
            annotations["omega_bin_map"] = omega_bin_map
            theta_bin_map = self.discretize_theta(theta_map, no_contact_mask)
            annotations["theta_bin_map"] = theta_bin_map
            phi_bin_map = self.discretize_phi(phi_map, no_contact_mask)
            annotations["phi_bin_map"] = phi_bin_map

            # filename
            _, filename = os.path.split(ann2d_path)
            filename_pre, ext = os.path.splitext(filename)
            annotations["filename"] = filename_pre



        if self.use_cache == 1 and self.cache["anns"].get(ann_paths["2d"]) is None:
            self.cache["anns"][ann_paths["2d"]] = {"contact_map": annotations["contact_map"]}
            #np.save(cache_path, {"contact": annotations["contact_map"]})
            with open(cache_path, 'wb') as handle:
                pickle.dump({"contact_map": annotations["contact_map"]}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("save {}".format(ann_paths))

        if ann_paths.get("profile") is not None:
            annotations["profile"] = np.load(ann_paths["profile"])
            if self.use_cache == 1:
                self.cache["anns"][ann_paths["2d"]]["profile"] = annotations["profile"]

        if ann_paths.get("pseudo_profile") is not None:
            annotations["pseudo_profile"] = np.load(ann_paths["pseudo_profile"])
            if self.use_cache == 1:
                self.cache["anns"][ann_paths["2d"]]["pseudo_profile"] = annotations["pseudo_profile"]


        return annotations

    def read_sequence(self, filename: str) -> Tuple[str, str]:
        """ Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))   # read the first line of MSA data
        return record.description, str(record.seq)

    def remove_insertions(self, sequence: str) -> str:
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(self.translation)

    def read_msa(self, filename: str, nseq: int, mode="top") -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        range_limit = nseq if mode == "top" else 20000
        if self.use_cache != True:
            full_record = []
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), range_limit):
                full_record.append((record.description, self.remove_insertions(str(record.seq))))
            self.cache["data"]["filename"] = full_record
        else:
            if self.cache["data"].get(filename) is None:
                full_record = []
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), range_limit):
                    full_record.append((record.description, self.remove_insertions(str(record.seq))))
                self.cache["data"]["filename"] = full_record
            else:
                full_record = self.cache["data"]["filename"]

        full_depth = len(full_record)

        if mode == "top":
            sampled_record = full_record

        elif mode == "random":
            sample_depth = nseq
            if full_depth > sample_depth:
                index_list = range(1, full_depth)
                downsample_index_list = random.sample(index_list, k=sample_depth - 1)
                downsample_index_list.sort()
                downsample_index_list = [0] + downsample_index_list

                sampled_record = []
                for index in downsample_index_list:
                    sampled_record.append(full_record[index])
            else:
                sampled_record = full_record
        else:
            raise Exception("Without this type {} of sampling MSA")

        return sampled_record


    #def read_msa(self, filename: str, nseq: int, mode: str) -> List[Tuple[str, str]]:
    #    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    #    return [(record.description, self.remove_insertions(str(record.seq)))
    #            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

    def read_2d_map(self, ann2d_path):
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
        dist_map = information_numpy[:, 2].reshape(cm_shape).astype(np.float)
        omega_map = information_numpy[:, 3].reshape(cm_shape).astype(np.float)
        theta_map = information_numpy[:, 4].reshape(cm_shape).astype(np.float)
        phi_map = information_numpy[:, 5].reshape(cm_shape).astype(np.float)

        return dist_map, omega_map, theta_map, phi_map

    def discretize_dist(self, dist_map):
        # build threshold list
        threshold_list = []
        threshold_num = 37
        for i in range(threshold_num):
            threshold_list.append(2 + i * 0.5)

        # discretization
        dist_bin_map = 0
        # index 1 - 36; otherwise index 0 = 0
        for i in range(threshold_num - 1):
            mask = (dist_map >= threshold_list[i]) & (dist_map < threshold_list[i+1])
            dist_bin_map = dist_bin_map + mask * (i + 1)

        # elements on the diagonal line and other disordered values
        #dist_bin_map = dist_bin_map + (dist_map < threshold_list[0]) * (-1)

        return dist_bin_map

    def discretize_omega(self, omega_map, no_contact_mask):
        """
        :param omega_map: -pi, +pi
        :param no_contact_mask: > 20 A.
        :return:
        """
        # convert radian to degree (0-360)
        omega_map = omega_map / np.pi * 0.5
        neg_mask = omega_map < 0
        omega_map[neg_mask] = 0.5 - omega_map[neg_mask]
        omega_map = omega_map * 360
        omega_map[omega_map == 360] = 0

        # build threshold list
        threshold_list = []
        threshold_num = 25
        for i in range(threshold_num):
            threshold_list.append(i * 15)

        # discretization
        omega_bin_map = 0
        for i in range(threshold_num-1):
            mask = (omega_map >= threshold_list[i]) & (omega_map < threshold_list[i+1])
            omega_bin_map = omega_bin_map + mask * (i + 1)
        omega_bin_map[no_contact_mask] = 0

        return omega_bin_map

    def discretize_theta(self, theta_map, no_contact_mask):
        """
        :param theta_map: -pi, +pi
        :param no_contact_mask: > 20 A.
        :return:
        """
        # convert radian to degree (0-360)
        theta_map = theta_map / np.pi * 0.5
        neg_mask = theta_map < 0
        theta_map[neg_mask] = 0.5 - theta_map[neg_mask]
        theta_map = theta_map * 360
        theta_map[theta_map == 360] = 0

        # build threshold list
        threshold_list = []
        threshold_num = 25
        for i in range(threshold_num):
            threshold_list.append(i * 15)

        # discretization
        theta_bin_map = 0
        for i in range(threshold_num - 1):
            mask = (theta_map >= threshold_list[i]) & (theta_map < threshold_list[i + 1])
            theta_bin_map = theta_bin_map + mask * (i + 1)
        theta_bin_map[no_contact_mask] = 0

        return theta_bin_map

    def discretize_phi(self, phi_map, no_contact_mask):
        """
        :param phi_map: -pi, +pi
        :param no_contact_mask: > 20 A.
        :return:
        """
        # convert radian to degree (0-180)
        phi_map = phi_map / np.pi * 180
        phi_map[phi_map == 180] = 0

        # build threshold list
        threshold_list = []
        threshold_num = 13
        for i in range(threshold_num):
            threshold_list.append(i * 15)

        # discretization
        phi_bin_map = 0
        for i in range(threshold_num - 1):
            mask = (phi_map >= threshold_list[i]) & (phi_map < threshold_list[i + 1])
            phi_bin_map = phi_bin_map + mask * (i + 1)
        phi_bin_map[no_contact_mask] = 0

        return phi_bin_map


    def read_sse(self, ann1d_path):
        ann1d_path = ann1d_path + ".sse"
        sse = []
        with open(ann1d_path, "r") as f:
            for line in f:
                if ">" in line:
                    continue
                for character in line.strip():
                    if character == "-":  # is that should continue?
                        sse_label = -1
                    else:
                        sse_label = self.class_to_label[character]
                    sse.append(sse_label)

        return sse