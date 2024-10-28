

import os
import time
import datetime
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
import glob

import datetime
import numpy as np
import pandas as pd
import cv2
import argparse
from setuptools._distutils.util import strtobool


import sys
sys.path.append('/userdir/im2rbte')
from augmentations import EdgeDetector, OriNMS, Thresholder, Cleaner
from utils import float_tuple
from PIL import Image

class COCOCropDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, crop_size=224, use_geometric=True, use_thinnms=True, use_hysteresis=True, use_large_components=True,):
        self.df = pd.read_csv(csv_path, header=None)
        self.dir = os.path.dirname(os.path.abspath(csv_path))
        self.pil_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),])
        self.affine = torchvision.transforms.RandomAffine(degrees=5, shear=0)
        # self.filps = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.RandomVerticalFlip(),
        # ])
        self.edge_d = EdgeDetector(edge_mode='normal')
        self.nms_model = cv2.ximgproc.createStructuredEdgeDetection('/userdir/Pretrained_Models/opencv_extra.yml.gz')
        self.nms = OriNMS(model=self.nms_model, prob=100, radious=2, bound_radious=0, multi=1.0)
        self.thresholder = Thresholder(thresh_rand=20, thresh_mode='normal', hyst_par=(0.5, 1.5), hyst_pert=0.2, hyst_prob=100, thinning=False)
        self.cleaner = Cleaner(percent_of_cc=(100, 100), del_less_than=(10, 10))

        self.random_resized_crop =  torchvision.transforms.RandomResizedCrop(size=crop_size, 
                                                    scale=float_tuple([0.8, 1.0]),
                                                    ratio=float_tuple([0.75, 1.3333333333333333]))
        trans_list = []
        
        if use_geometric:
            trans_list.extend([self.affine, self.random_resized_crop, torchvision.transforms.RandomHorizontalFlip()])
        else :
            trans_list.append(torchvision.transforms.Resize(crop_size))

        trans_list.append(self.edge_d)

        if use_thinnms:
            trans_list.append(self.nms)
        if use_hysteresis:
            trans_list.append(self.thresholder)
        if use_large_components:
            trans_list.append(self.cleaner)
        
        trans_list.append(torchvision.transforms.ToTensor())

        self.trans = torchvision.transforms.Compose(trans_list)
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        # image_path = os.path.join(self.dir, self.df[0][idx])
        edge_path = os.path.join(self.dir, self.df[1][idx])
        label = int(self.df[2][idx])
        
        image = Image.open(edge_path).convert('RGB') # cv2.imread(edge_path).astype(np.float32)
        image = self.trans(image).to(torch.float32)

        # color_image = Image.open(image_path).convert('RGB')
        # color_image = self.pil_to_tensor(color_image)

        # image = torch.cat((color_image, edge_image), dim=0)
        # image = self.filps(image)
        return image, label



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('--traincsv', type=str, default='dataset/train.csv')
    parser.add_argument('--valcsv', type=str, default='dataset/val.csv')
    parser.add_argument('--testdir', type=str, default='dataset/test')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--classnum', type=int, default=2)
    parser.add_argument('--use_geometric', type=strtobool, default=1)
    parser.add_argument('--use_thinnms', type=strtobool, default=1)
    parser.add_argument('--inputsize', type=int, default=224)
    
    args = parser.parse_args()

    device = args.device
    num_class = args.classnum
    
    if args.output == '':
        date_string = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        outputdir = 'output-{}-{}'.format(date_string, args.model)
    else :
        outputdir = args.output

    
    
    if args.model.find('resnet') != -1:
        model = torchvision.models.get_model(args.model, weights="DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_class)
    else :
        import timm
        model = timm.create_model(args.model, pretrained=True, num_classes=num_class)
    model.to(device)

    use_geometric = args.use_geometric==1
    use_thinnms = args.use_thinnms==1
    use_hysteresis = args.use_thinnms==1
    use_large_components = args.use_thinnms==1

    trainset = COCOCropDataset(args.traincsv, use_geometric=use_geometric,  use_thinnms= use_thinnms, use_hysteresis=use_hysteresis,use_large_components=use_large_components, crop_size=args.inputsize)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = args.batchsize, shuffle = True, num_workers = 8)
    
    valset = COCOCropDataset(args.valcsv, use_geometric=use_geometric,  use_thinnms= use_thinnms, use_hysteresis=use_hysteresis,use_large_components=use_large_components, crop_size=args.inputsize)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 1)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr)])

    # 出力先ディレクトリ作成
    os.makedirs(outputdir, exist_ok=False)
    
    # logger準備
    writer = SummaryWriter(log_dir=outputdir)
    itr_num = 0

    with tqdm(range(args.epoch)) as pbar_epoch:
        for ep in pbar_epoch:
            pbar_epoch.set_description("[Epoch %d]" % (ep+1))
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False) as pbar_itr:
                for i, (images, labels) in pbar_itr:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    pred = model(images)
                    loss = loss_func(pred, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss_value = loss.cpu().item()
                    
                    writer.add_scalar("training_loss", train_loss_value, itr_num)
                    pbar_itr.set_postfix(OrderedDict(training_loss=train_loss_value))


                    if itr_num % 100 == 0:
                        writer.add_images("train_example", images, itr_num, dataformats='NCHW')

                    itr_num += 1
            model.eval()
            acc_list = []
            loss_list = []
            for i, (images, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    pred = model(images)
                    loss_list.append(loss_func(pred, labels).cpu().item())
                    for w in (torch.argmax(pred, dim=1)==labels).detach().cpu().numpy():
                        acc_list.append(w)
            writer.add_scalar("validation/accuracy", np.mean(acc_list), ep+1)
            writer.add_scalar("validation/loss", np.mean(loss_list), ep+1)
            model.train()
            # torch.save(model, os.path.join(outputdir, "./model_{0:03d}.pth".format(ep+1)))
    torch.save(model, os.path.join(outputdir, "./model_final.pth"))
    print()
    for testfile in sorted(glob.glob(os.path.join(args.testdir, '*png'))+glob.glob(os.path.join(args.testdir, '*jpeg'))+glob.glob(os.path.join(args.testdir, '*jpg'))):
        test_img = cv2.imread(testfile)
        if np.mean(test_img) > 128:
            test_img = 255-test_img
        test_input = torch.Tensor(np.transpose(cv2.resize(test_img, (args.inputsize, args.inputsize))/255, (2,0,1))[np.newaxis]).to(device)
        print(testfile, model(test_input).detach().cpu().softmax(dim=1).numpy())