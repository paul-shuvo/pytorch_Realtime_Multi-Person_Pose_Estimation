import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp

from freenect2 import Device, FrameType


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)  
print(torch.cuda.get_device_name(0)) 
# assert torch.cuda.get_device_name(0) is 'GeForce GTX 1080'
model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model.cuda()
model.float()
model.eval()

device = Device()
device.start()
t = True
if __name__ == "__main__":
    
    # video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        if t:
            t = False
            time.sleep(2)
        type_, frame = device.get_next_frame()
        if type_ is FrameType.Color: # FrameFormat.BGRX
            rgb = frame.to_array().astype(np.uint8)[:,:,0:3]
            # cv2.imshow('rgb', rgb[:,:,0:3])
            oriImg = rgb
        else:
            continue
        
        shape_dst = np.min(oriImg.shape[0:2])

        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')
                  
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
                
        out = draw_humans(oriImg, humans, imgcopy=True)
        # print(type(out))

        # Display the resulting frame
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            device.stop()
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
