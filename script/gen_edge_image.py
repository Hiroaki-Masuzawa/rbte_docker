import sys
import os
sys.path.append(os.path.dirname(__file__))

from edge_detection import detect_SE_edge, detect_BDCN_edge, detect_hed_edge, get_SE_model, get_BDCN_model, get_hed_model

import argparse
from setuptools._distutils.util import strtobool

import numpy as np
import pandas as pd
import json

import time

import cv2
import torch
from torch.nn import functional as F
import torchvision


def gen_edge_image(image_bgr, device='cuda'):
    se_result = detect_SE_edge(se_model, image_bgr)
    bdcn_result = detect_BDCN_edge(bdcn_model, image_bgr, device)
    hed_result = detect_hed_edge(hed_model, image_bgr, device)
    edge_output = (np.stack([bdcn_result, hed_result, se_result]).transpose(1,2,0)*255).astype(np.uint8)
    return edge_output
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('input', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--data_type', type=str, choices=['train', 'val'])
    parser.add_argument('--data_root_dir', type=str, default='dataset')
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()


    device = args.device

    se_model = get_SE_model()

    bdcn_model = get_BDCN_model()
    bdcn_model.to(device).eval()

    hed_model =  get_hed_model()
    hed_model.to(device).eval()
    
    if args.input.split('.')[-1] == 'csv':
        df = pd.read_csv(args.input, header=None)
        dir_name = os.path.dirname(args.input)
        for i in range(df.shape[0]):
            image_file = os.path.join(dir_name, df[0][i])
            output_file = os.path.join(dir_name, df[0][i].replace('image', 'edge'))
            print(image_file, output_file)
            image_bgr = cv2.imread(image_file)
            edge_output = gen_edge_image(image_bgr, device)
            cv2.imwrite(output_file, edge_output)
    elif args.input.split('.')[-1] == 'json':
        image_size = (256, 256)
        # target_ids = None
        target_ids = [53, 55]
        padding_ratio = 0.1
        keep_aspect = True

        with open(args.input, 'r') as f:
            dat = json.load(f)
        annos = [anno for anno in dat["annotations"] if anno['bbox'][2] >= image_size[0]//2 and anno['bbox'][3] >= image_size[1]//2 and anno['iscrowd'] != 1 and anno['category_id'] in target_ids]

        if target_ids is None:
            map_label = {c:i for i, c in enumerate(set([anno["category_id"] for anno in dat['annotations']]))}
        else :
            map_label = {c:i for i, c in enumerate(target_ids)}

        os.makedirs(os.path.join(args.data_root_dir, args.data_type), exist_ok=True)

        for anno in annos:
            image_id = anno['image_id']
            image_dat = [img for img in dat['images'] if img['id']==image_id][0]
            image_path = os.path.join(args.input_dir, image_dat['file_name'])
            bbox = np.array(anno['bbox'])

            xs = int(bbox[0]-bbox[2]*padding_ratio)
            xe = int(bbox[0]+bbox[2]+bbox[2]*padding_ratio)
            ys = int(bbox[1]-bbox[3]*padding_ratio)
            ye = int(bbox[1]+bbox[3]+bbox[3]*padding_ratio)
            if keep_aspect:
                new_width = xe-xs
                new_height = ye-ys
                if new_width > new_height:
                    diff = new_width-new_height
                    ys -= diff//2
                    ye += diff//2
                else :
                    diff = new_height-new_width
                    xs -= diff//2
                    xe += diff//2
            xs = max(0, xs)
            xe = min(image_dat['width']-1, xe)
            ys = max(0, ys)
            ye = min(image_dat['height']-1, ye)
            image = cv2.imread(image_path)
            crop_image = image[ys:ye, xs:xe]
            crop_image = cv2.resize(crop_image, image_size)
            edge_output = gen_edge_image(crop_image, device)
            image_output_file = os.path.join(args.data_type, "image_{}.png".format(anno['id']))
            edge_output_file = os.path.join(args.data_type, "edge_{}.png".format(anno['id']))
            cv2.imwrite(os.path.join(args.data_root_dir, image_output_file), crop_image)
            cv2.imwrite(os.path.join(args.data_root_dir, edge_output_file), edge_output)
            print("{},{},{}".format(image_output_file, edge_output_file, map_label[anno["category_id"]]))
            
            
        