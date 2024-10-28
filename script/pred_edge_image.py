
import os
import argparse
import glob
import numpy as np

import cv2

import torch

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('--testdir', type=str, default='dataset/test')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--modelfile', type=str)
    parser.add_argument('--inputsize', type=int, default=224)
    
    args = parser.parse_args()

    device = args.device
    model = torch.load(args.modelfile, map_location=device)
    for testfile in sorted(glob.glob(os.path.join(args.testdir, '*png'))+glob.glob(os.path.join(args.testdir, '*jpeg'))+glob.glob(os.path.join(args.testdir, '*jpg'))):
        test_img = cv2.imread(testfile)
        if np.mean(test_img) > 128:
            test_img = 255-test_img
        test_input = torch.Tensor(np.transpose(cv2.resize(test_img, (args.inputsize, args.inputsize))/255, (2,0,1))[np.newaxis]).to(device)
        print(testfile, model(test_input).detach().cpu().softmax(dim=1).numpy())    
