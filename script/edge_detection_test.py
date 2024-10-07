import sys
import os
sys.path.append(os.path.dirname(__file__))

from edge_detection import detect_SE_edge, detect_BDCN_edge, detect_hed_edge, get_BDCN_model, get_hed_model

sys.path.append('/userdir/im2rbte')
from augmentations import EdgeDetector, OriNMS, Thresholder, Cleaner

import argparse
from setuptools._distutils.util import strtobool

import numpy as np

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
    parser.add_argument('--input', type=str, default="/userdir/images/color/Lenna.bmp")
    parser.add_argument('--output', type=str, default="output.png")
    parser.add_argument('--debug_image', type=str, default="result.png")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--benchmark', type=strtobool, default=0)
    args = parser.parse_args()

    device = args.device

    image_bgr = cv2.imread(args.input)
    
    bdcn_model = get_BDCN_model()
    bdcn_model.to(device).eval()

    hed_model =  get_hed_model()
    hed_model.to(device).eval()

    edge_d = EdgeDetector(edge_mode='normal')
    nms_model = cv2.ximgproc.createStructuredEdgeDetection('/userdir/Pretrained_Models/opencv_extra.yml.gz')
    nms = OriNMS(model=nms_model, prob=100, radious=2, bound_radious=0, multi=1.0)
    thresholder = Thresholder(thresh_rand=20, thresh_mode='normal', hyst_par=(0.5, 1.5), hyst_pert=0.2, hyst_prob=100, thinning=False)
    cleaner = Cleaner(percent_of_cc=(100, 100), del_less_than=(10, 10))
    
    if args.benchmark == 1:
        _ = detect_SE_edge(image_bgr)
        _ = detect_BDCN_edge(bdcn_model, image_bgr, device)
        _ = detect_hed_edge(hed_model, image_bgr, device)

        se_time = []
        bdcn_time = []
        hed_time = []
        edgedetect_time = []
        nms_time = []
        thresholder_time = []
        cleaner_time = []

        for _ in range(100):
            start_time = time.time()
            se_result = detect_SE_edge(image_bgr)
            se_end = time.time()
            bdcn_result = detect_BDCN_edge(bdcn_model, image_bgr, device)
            bdcn_end = time.time()
            hed_result = detect_hed_edge(hed_model, image_bgr, device)
            hed_end = time.time()
            edge_output = (np.stack([bdcn_result, hed_result, se_result]).transpose(1,2,0)*255).astype(np.uint8)
            se_time.append(se_end-start_time)
            bdcn_time.append(bdcn_end-se_end)
            hed_time.append(hed_end-bdcn_end)

            samples = []
            gen_num = 3
            for _ in range(gen_num):
                trans_start = time.time()
                work1 = edge_d(edge_output)
                edgedetect_end = time.time()
                work2 = nms(work1)
                nms_end = time.time()
                work3 = thresholder(work2)
                thresholder_end = time.time()
                work4 = cleaner(work3)
                cleaner_end = time.time()
                work = np.concatenate([work1, work2, work3, work4], axis=1)
                # work = cv2.hconcat([edge_output, (work*255).astype(np.uint8)])
                # samples.append(work)
                samples.append(work)
                edgedetect_time.append(edgedetect_end-trans_start)
                nms_time.append(nms_end-edgedetect_end)
                thresholder_time.append(thresholder_end-nms_end)
                cleaner_time.append(cleaner_end-thresholder_end)

            end_time = time.time()

            samples2 = [cv2.hconcat([edge_output, (work*255).astype(np.uint8)]) for work in samples]
            sample = cv2.vconcat(samples2)
            
            cv2.imwrite("work.png", sample)
            se_result_v = cv2.cvtColor((se_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            bdcn_result_v = cv2.cvtColor((bdcn_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            hed_result_v = cv2.cvtColor((hed_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            show_img = cv2.hconcat([image_bgr, bdcn_result_v, hed_result_v, se_result_v])
            cv2.imwrite(args.debug_image, show_img)
            cv2.imwrite(args.output, edge_output)

        # print(end_time-start_time)
        print("SE", np.mean(se_time)*1000)
        print("BDCN", np.mean(bdcn_time)*1000)
        print("HED", np.mean(hed_time)*1000)
        print("DETECT", np.mean(edgedetect_time)*1000)
        print("NMS", np.mean(nms_time)*1000)
        print("THRESHOLDER", np.mean(thresholder_time)*1000)
        print("CLANER", np.mean(cleaner_time)*1000)
    else :
        se_result = detect_SE_edge(image_bgr)
        bdcn_result = detect_BDCN_edge(bdcn_model, image_bgr, device)
        hed_result = detect_hed_edge(hed_model, image_bgr, device)
        se_result_v = cv2.cvtColor((se_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        bdcn_result_v = cv2.cvtColor((bdcn_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        hed_result_v = cv2.cvtColor((hed_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        show_img = cv2.hconcat([image_bgr, bdcn_result_v, hed_result_v, se_result_v])
        output = (np.stack([bdcn_result, hed_result, se_result]).transpose(1,2,0)*255).astype(np.uint8)
        cv2.imwrite(args.debug_image, show_img)
        cv2.imwrite(args.output, output)

        edge_d = EdgeDetector(edge_mode='normal')
        nms_model = cv2.ximgproc.createStructuredEdgeDetection('/userdir/Pretrained_Models/opencv_extra.yml.gz')
        nms = OriNMS(model=nms_model, prob=100, radious=2, bound_radious=0, multi=1.0)
        thresholder = Thresholder(thresh_rand=20, thresh_mode='normal', hyst_par=(0.5, 1.5), hyst_pert=0.2, hyst_prob=100, thinning=False)
        cleaner = Cleaner(percent_of_cc=(100, 100), del_less_than=(10, 10))
        tr_list = [edge_d, nms, thresholder, cleaner]
        trans = torchvision.transforms.Compose(tr_list)
        samples = []
        for _ in range(3):
            if True:
                start_time = time.time()
                work1 = edge_d(output)
                work2 = nms(work1)
                work3 = thresholder(work2)
                work4 = cleaner(work3)
                end_time = time.time()
                print(end_time-start_time)
                work = np.concatenate([work1, work2, work3, work4], axis=1)
                work = cv2.hconcat([output, (work*255).astype(np.uint8)])
            else :
                start_time = time.time()
                work = trans(output)
                end_time = time.time()
                print(end_time-start_time)
                work = cv2.hconcat([output, (work*255).astype(np.uint8)])
            samples.append(work)
        sample = cv2.vconcat(samples)
        cv2.imwrite("work.png", sample)
