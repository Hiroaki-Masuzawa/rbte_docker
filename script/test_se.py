import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread('/userdir/images/color/Lenna.bmp')
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
    imgrgb = imgrgb.astype(np.float32)
    model = '/userdir/se_model/model.yml'
    retval = cv2.ximgproc.createStructuredEdgeDetection(model)
    out = retval.detectEdges(imgrgb)
    cv2.imwrite("/userdir/result_se.png", (out*255).astype(np.uint8))