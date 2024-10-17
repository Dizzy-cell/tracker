
from .model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

class FaceParse:
    def __init__(self, pretrained_model = '79999_iter.pth'):
        
        self.atts = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.n_classes = 19
        self.net = BiSeNet(n_classes = 19)
        self.net.load_state_dict(torch.load(pretrained_model))
        self.net.cuda()
        
        self.net.eval()
        
        self.image_size = (512, 512)
        self.to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    def solveImageFile(self, image_path = 'test.png', mask_path = 'mask.png'):
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize(self.image_size, Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0).cuda()
            out = self.net(img)[0]
            
            parsing = out[0].cpu().numpy().argmax(0)

            mask = self.makemask(parsing)
            mask = mask.astype(np.uint8)
            cv2.imwrite(mask_path, mask)

    def makemask(self, parsing):
        
        part_colors = [[255, 255, 255] for _ in range(19)]
        part_colors[0] = [0,0,0]
        part_colors[16] = [0,0,0]
        
        part_colors[15]=[0,0,0]
        part_colors[14]=[0,0,0]
        
        mask = np.zeros((parsing.shape[0], parsing.shape[1], 3))
        for pi in range(1, self.n_classes):
            index = np.where(parsing == pi)
            mask[index[0], index[1], :] = part_colors[pi]
        
        return mask
