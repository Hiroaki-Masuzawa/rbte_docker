import cv2
import torch
import torchvision
import sys
import numpy as np

sys.path.append('/userdir/BDCN')
import bdcn

sys.path.append('/userdir/hed')
from hed_network import Network

def get_SE_model():
    model = '/userdir/se_model/model.yml'
    retval = cv2.ximgproc.createStructuredEdgeDetection(model)
    return retval

def detect_SE_edge(model, image):
    imgrgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)/255
    imgrgb = imgrgb.astype(np.float32)
    # model = '/userdir/se_model/model.yml'
    # retval = cv2.ximgproc.createStructuredEdgeDetection(model)
    out = model.detectEdges(imgrgb)
    return out

def get_BDCN_model():
    bdcn_model = bdcn.BDCN()
    bdcn_model.load_state_dict(torch.load('/userdir/bdcn_model/final-model/bdcn_pretrained_on_bsds500.pth'))
    return bdcn_model

def detect_BDCN_edge(model, image, device):
    data = image.astype(np.float32)
    data -= np.array([[[104.00699, 116.66877, 122.67892]]])
    data = torch.Tensor(data[np.newaxis,:,:,:].transpose(0,3,1,2)).to(device)
    with torch.no_grad():
        out = model(data)
    # fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    return fuse

def get_hed_model():
    hed_model = Network()
    return hed_model

def detect_hed_edge(model, image, device):
    raw_height, raw_width = image.shape[0:2]

    image = cv2.resize(image, (480,320))

    tenInput = torch.FloatTensor(np.ascontiguousarray(np.array(image).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    # model = model.cuda().eval()
    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    tenInput = tenInput.to(device)
    with torch.no_grad():
        tenOutput = model(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :]
    tenOutput = torchvision.transforms.functional.resize(img=tenOutput, size=(raw_height, raw_width))
    return tenOutput.clip(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
