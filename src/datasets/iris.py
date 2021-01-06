#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import os
import random
import re

from torch.utils.data import dataset, sampler
from torchvision.datasets.folder import default_loader
from datasets.base_dataset import BaseDataset


class IRIS(BaseDataset):
    """
    Attributes:
        imgs (list of str): dataset image file paths
        _id2label (dict): mapping from person id to softmax continuous label
    """

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        id=file_path.split('/')[-1].split('_')[0]
        if "L" in id:
            id=id.replace("L", "1")
        else:
            id=id.replace("R", "2")
        return int(id)

