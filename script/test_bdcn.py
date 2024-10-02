import sys

import numpy as np

import cv2
import torch
from torch.nn import functional as F

sys.path.append('/userdir/BDCN')
import bdcn


if __name__ == '__main__':
    model = bdcn.BDCN()
    model.load_state_dict(torch.load('/userdir/bdcn_model/final-model/bdcn_pretrained_on_bsds500.pth'))
    model.eval()
    data = cv2.imread('/userdir/images/color/Lenna.bmp').astype(np.float32)
    data -= np.array([[[104.00699, 116.66877, 122.67892]]])
    # print(data.shape)
    data = torch.Tensor(data[np.newaxis,:,:,:].transpose(0,3,1,2))
    # print(data.shape)
    with torch.no_grad():
        out = model(data)
    fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    print(fuse.shape)
    cv2.imwrite('result_bdcn.png', fuse*255)