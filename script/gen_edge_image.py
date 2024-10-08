import sys
import os
sys.path.append(os.path.dirname(__file__))

from edge_detection import detect_SE_edge, detect_BDCN_edge, detect_hed_edge, get_SE_model, get_BDCN_model, get_hed_model

import argparse
from setuptools._distutils.util import strtobool

import numpy as np
import pandas as pd

import time

import cv2
import torch
from torch.nn import functional as F
import torchvision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('inputcsv', type=str)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    df = pd.read_csv(args.inputcsv, header=None)
    dir_name = os.path.dirname(args.inputcsv)

    device = args.device

    se_model = get_SE_model()

    bdcn_model = get_BDCN_model()
    bdcn_model.to(device).eval()

    hed_model =  get_hed_model()
    hed_model.to(device).eval()
    

    for i in range(df.shape[0]):
        image_file = os.path.join(dir_name, df[0][i])
        output_file = os.path.join(dir_name, df[0][i].replace('image', 'edge'))
        print(image_file, output_file)
        image_bgr = cv2.imread(image_file)
        se_result = detect_SE_edge(se_model, image_bgr)
        bdcn_result = detect_BDCN_edge(bdcn_model, image_bgr, device)
        hed_result = detect_hed_edge(hed_model, image_bgr, device)
        edge_output = (np.stack([bdcn_result, hed_result, se_result]).transpose(1,2,0)*255).astype(np.uint8)
        cv2.imwrite(output_file, edge_output)