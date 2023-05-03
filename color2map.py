import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import PIL.Image as Image
from torchvision import transforms as TR
import random
import numpy as np
import argparse
import os
import pickle
from PIL import Image
from tqdm import tqdm
import imageio
import cv2
import math
import torch.nn.utils.spectral_norm as sp
import torchvision.transforms as tf
from scipy.sparse import dok_matrix
from pathlib import Path
from scipy import linalg
import json
color_dict = {}
color_to_id = {}
color_to_trainid = {}
path = r"C:\Users\guodi\Desktop\mapillary\config_v2.0.json"
color_path = r"C:\Users\guodi\Desktop\mapillary\training\v2.0\labels"
id_output_path = r"C:\Users\guodi\Desktop\output\id"
trainid_output_path = r"C:\Users\guodi\Desktop\output\trainid"

with open(path) as f:
    config = json.load(f)
config_labels = config["labels"]
for item in config_labels:
    color_dict[str(item["color"])] = item["readable"]

cityscapes_ids = [
    #       name              id    trainId   category            catId     hasInstances   ignoreInEval   color
    ['unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (0, 0, 0),],
    ['ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (0, 0, 0),],
    ['rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (0, 0, 0),],
    ['out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (0, 0, 0),],
    ['static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (0, 0, 0),],
    ['dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74, 0),],
    ['ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , (81, 0, 81),],
    ['road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64, 128),],
    ['sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35, 232),],
    ['parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250, 170, 160),],
    ['rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230, 150, 140),],
    ['building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , (70, 70, 70),],
    ['wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102, 102, 156),],
    ['fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190, 153, 153),],
    ['guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180, 165, 180),],
    ['bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150, 100, 100),],
    ['tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150, 120, 90),],
    ['pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153, 153, 153),],
    ['polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153, 153, 153),],
    ['traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250, 170, 30),],
    ['traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220, 220, 0),],
    ['vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107, 142, 35),],
    ['terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152, 251, 152),],
    ['sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , (70, 130, 180),],
    ['person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60),],
    ['rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255, 0, 0),],
    ['car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (0, 0, 142),],
    ['truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (0, 0, 70),],
    ['bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (0, 60, 100),],
    ['caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (0, 0, 90),],
    ['trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (0, 0, 110),],
    ['train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (0, 80, 100),],
    ['motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (0, 0, 230),],
    ['bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32),],
]

for i in cityscapes_ids:
    a = str(i[7]).replace("(", "[").replace(")", "]")
    if a in color_dict.keys():
        color_to_id[str(i[7]).replace("(", "[").replace(")", "]")] = i[1]
        color_to_trainid[str(i[7]).replace("(", "[").replace(")", "]")] = i[2]


def generate_id_maps(color_dict, color_to_id, color_path, output_path):
    for color in color_dict.keys():
        if color in color_to_id.keys():
            color_dict[color] = color_to_id[color]
        else:
            color_dict[color] = 0

    for file in os.listdir(color_path):

        img = Image.open(os.path.join(color_path, file)).convert("RGB").resize([512, 256], resample=Image.Resampling.NEAREST)
        img = np.array(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # print(img[i][j])

                a = "[" + str(img[i][j][0]) + ", " + str(img[i][j][1]) + ", " + str(img[i][j][2]) + "]"
                img[i][j][0] = color_dict[a]
                # if img[i][j][0] != 0:
                #     img[i][j][0] = 255

        img = img[:, :, 0]
        print(file)
        print(np.unique(img))

        cv2.imwrite(os.path.join(output_path, file), img)




# def generate_trainid_maps(color_dict, color_to_trainid):
#     for color in color_dict.keys():
#         if color in color_to_trainid.keys():
#             color_dict[color] = color_to_trainid[color]
#         else:
#             color_dict[color] = 255


# generate_id_maps(color_dict, color_to_id, color_path, id_output_path)
generate_id_maps(color_dict, color_to_id, color_path, id_output_path)














