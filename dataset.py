# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import os
join = os.path.join
from skimage import io, segmentation, morphology, exposure, measure
import numpy as np
import tifffile as tif
import argparse
from stardist import star_dist,edt_prob

import torchvision

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous
from monai.utils import MAX_SEED, deprecated_arg, get_seed, look_up_option, min_version, optional_import
from monai.utils.misc import first
from monai.data import decollate_batch, PILReader
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Resized,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    # RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    # RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,    
    EnsureTyped,
    EnsureType,
)
if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")

def random_resize(pack):
    _, h, w = pack["img"].shape
    rand_rate = (1.2 - 0.8) * np.random.random_sample() + 0.8
    H = np.int64(h * rand_rate)
    W = np.int64(w * rand_rate)
    resize = Resized(keys=["img", "img_strong", "label_blood", "label_choroid"], spatial_size=(H, W), size_mode='all', mode=['bilinear', 'bilinear', 'nearest', 'nearest'], align_corners=[False, False, None, None], allow_missing_keys=True)
    pack = resize(pack)
    return pack


def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior


def create_bboxes(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    object_num = len(measure.regionprops(inst_map))
    boxes = np.zeros( (object_num, 4) )
    masks = np.zeros((object_num, inst_map.shape[0], inst_map.shape[1]))
    #print(np.unique(inst_map))
    #print(object_num, len(np.unique(inst_map)))
    for i, region in enumerate(measure.regionprops(inst_map)):
        boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3] = region.bbox
        ins_label = inst_map[region.coords[0, 0], region.coords[0, 1]]
        masks[i, :, :] = np.uint8( (inst_map == ins_label) )
        #print(inst_map[region.coords[0, 0], region.coords[0, 1]], inst_map[region.coords[1, 0], region.coords[1, 1]])
    return boxes, masks


class Dataset(_TorchDataset):
    
    def __init__(self, data: Sequence, mode="train", input_size=256, use_resize=-1, use_cp=-1, path_index: Optional[Callable] = None) -> None:

        self.data = data
        self.mode = mode
        self.transforms_base = Compose(
        [
            LoadImaged(keys=["img", "label_blood", "label_choroid"], reader=PILReader, dtype=np.uint8, allow_missing_keys=True), # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["img", "label_blood", "label_choroid"], allow_missing_keys=True), # label: (1, H, W)
        ]
        )
        self.transforms_weak = Compose(
        [
            ScaleIntensityd(keys=["img", "img_strong"], allow_missing_keys=True), # Do not scale label
            SpatialPadd(keys=["img", "img_strong", "label_blood", "label_choroid"], spatial_size=(input_size, input_size), allow_missing_keys=True),
            RandSpatialCropd(keys=["img", "img_strong", "label_blood", "label_choroid"], roi_size=(input_size, input_size), random_size=False, allow_missing_keys=True),
            #RandAxisFlipd(keys=["img", "img_strong", "label_blood", "label_choroid"], prob=0.5, allow_missing_keys=True),
            #RandRotate90d(keys=["img", "img_strong", "label_blood", "label_choroid"], prob=0.5, spatial_axes=[0, 1], allow_missing_keys=True),  
            RandZoomd(keys=["img", "img_strong" , "label_blood", "label_choroid"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=['area', 'area', 'nearest', 'nearest'], allow_missing_keys=True),
            EnsureTyped(keys=["img", "img_strong", "label_blood", "label_choroid"], allow_missing_keys=True),      
        ]
        )

        self.transforms_strong = Compose(
        [
            RandGaussianNoised(keys=['img'], prob=0.25, mean=0, std=0.1, allow_missing_keys=True),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1,2), allow_missing_keys=True),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1,2), allow_missing_keys=True),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3, allow_missing_keys=True),
        ]
        )

        self.path_index = path_index
        self.use_cp = use_cp
        self.use_resize = use_resize

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        data_i = self.data[index]

        #####Adopt Base Transform
        packs = apply_transform(self.transforms_base, data_i)
        packs["img"] = packs["img"].transpose(0, 2, 1)

        if self.mode == "train": ##AF B-Scans
            packs["label_blood"] = packs["label_blood"].transpose(0, 2, 1)
            packs["label_choroid"] = packs["label_choroid"].transpose(0, 2, 1)

            if self.use_resize == 1:
                packs = random_resize(packs)
            packs = apply_transform(self.transforms_weak, packs)
            packs["label_blood"] = packs["label_blood"] * packs["label_choroid"]

            return packs

        elif self.mode == "semi":  ##RF B-Scans
            
            ##Strong Augmentation
            packs_strong = apply_transform(self.transforms_strong, packs.copy())
            packs["img_strong"] = packs_strong["img"].copy()
            if self.use_resize == 1:
                packs = random_resize(packs)
            ##Weak Augmentation
            packs = apply_transform(self.transforms_weak, packs)

            return packs
        else: #validation
            packs = ScaleIntensityd(keys=["img", "img_strong"], allow_missing_keys=True)(packs)
            if "label_blood" in packs.keys():
                packs["label_blood"] = packs["label_blood"].transpose(0, 2, 1)
            if "label_choroid" in packs.keys():
                packs["label_choroid"] = packs["label_choroid"].transpose(0, 2, 1)
            ensure = EnsureTyped(keys=["img", "img_strong", "label_blood", "label_choroid"], allow_missing_keys=True)
            packs = ensure(packs)
            if "label_choroid" in packs.keys() and "label_blood" in packs.keys():
                packs["label_blood"] = packs["label_blood"] * packs["label_choroid"]
            #packs["img"] = packs["img"].expand(3, packs["img"].shape[1], packs["img"].shape[2])
            
            return packs