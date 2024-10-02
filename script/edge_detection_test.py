import sys

import numpy as np

import cv2
import torch
from torch.nn import functional as F
import torchvision

sys.path.append('/userdir/BDCN')
import bdcn

sys.path.append('/userdir/hed')
from run import Network


def detect_SE_edge(image):
    imgrgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)/255
    imgrgb = imgrgb.astype(np.float32)
    model = '/userdir/se_model/model.yml'
    retval = cv2.ximgproc.createStructuredEdgeDetection(model)
    out = retval.detectEdges(imgrgb)
    return out

def detect_BDCN_edge(model, image):
    data = image.astype(np.float32)
    data -= np.array([[[104.00699, 116.66877, 122.67892]]])
    data = torch.Tensor(data[np.newaxis,:,:,:].transpose(0,3,1,2))
    with torch.no_grad():
        out = model(data)
    fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    return fuse

def detect_hed_edge(model, image):
    raw_height, raw_width = image.shape[0:2]

    image = cv2.resize(image, (480,320))

    tenInput = torch.FloatTensor(np.ascontiguousarray(np.array(image).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    # model = model.cuda().eval()
    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenOutput = model(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :]
    tenOutput = torchvision.transforms.functional.resize(img=tenOutput, size=(raw_height, raw_width))
    return tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0]
if __name__ == '__main__':
    image_bgr = cv2.imread('/userdir/images/color/Lenna.bmp')
    
    
    bdcn_model = bdcn.BDCN()
    bdcn_model.load_state_dict(torch.load('/userdir/bdcn_model/final-model/bdcn_pretrained_on_bsds500.pth'))
    bdcn_model.eval()

    hed_model = Network()
    hed_model.eval()
    
    se_result = detect_SE_edge(image_bgr)
    bdcn_result = detect_BDCN_edge(bdcn_model, image_bgr)
    hed_result = detect_hed_edge(hed_model, image_bgr)
    se_result_v = cv2.cvtColor((se_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    bdcn_result_v = cv2.cvtColor((bdcn_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    hed_result_v = cv2.cvtColor((hed_result*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    show_img = cv2.hconcat([image_bgr, se_result_v, bdcn_result_v, hed_result_v])
    cv2.imwrite('result.png', show_img)