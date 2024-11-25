 # -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import os
import pickle

import numpy as np
# Modified from smplx code for FLAME
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from skimage.io import imread
from loguru import logger

from flame.lbs import lbs

from yacs.config import CfgNode as CN

from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj, load_obj

from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights
)

from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams

cfg = CN()
cfg.ict_face = 'face.npy'
cfg.ict_exp_model = 'blend_exp.npz'

cfg.bs = './flame/bs/exp/'

blend_name = ['Basis', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 
              'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
              'jawRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthPucker', 'mouthShrugLower', 'mouthShrugUpper', 'noseSneerLeft', 'noseSneerRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthLeft', 'mouthRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthSmileLeft', 'mouthSmileRight', 'tongueOut', 'eyeBlinkLeft', 'eyeBlinkRight']

media_name = ['Basis', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 
              'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
              'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 
              'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 
              'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight', 
              'tongueOut']


I = matrix_to_rotation_6d(torch.eye(3)[None].cuda())

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class Blend(nn.Module):
    def __init__(self, config, actor_name='actor'):
        super(Blend, self).__init__()
        logger.info(f"[Blend] Creating the Blend model")
        
        exp_face = np.zeros((5023, 3, 51))
        canonical = np.zeros((5023, 3))
        
        verts, faces, aux = load_obj(config.save_folder + actor_name + '/canonical.obj')
        canonical = verts.numpy()
        self.faces = faces.verts_idx.numpy()
        
        # print('Blend 106')
        # from IPython import embed  
        # embed()
        
       # basis, _ , _ = 
        
        for i, name in enumerate(media_name[1:-1]):
            verts, faces, aux = load_obj( cfg.bs + name + '.obj')
            exp_face[:,:, i]= verts.numpy() - canonical
        
        self.register_parameter('exp_face', nn.Parameter(to_tensor(exp_face), requires_grad=True))
        self.register_parameter('canonical', nn.Parameter(to_tensor(canonical), requires_grad=True))
        
    
    def forward(self, exp_parms):
        batch_size = exp_parms.shape[0]
        exp_face  = self.exp_face.expand(batch_size, -1, -1, -1)
        exp_parms = exp_parms.unsqueeze(dim = 1).unsqueeze(dim = 2)
        out = (exp_face * exp_parms).sum(axis = 3)
        out = self.canonical.expand(batch_size, -1, -1) + out
        
        return out

    def saveBlend(self, vertices = None, save_file = 'BlendMesh.obj'):
        if vertices is None:
            vertices = self.canonical
        faces = torch.from_numpy(self.faces).to(self.canonical.device)
        save_obj(save_file, vertices, faces)
        return

    def saveData(self, save_file = 'BlendModel.npz'):
        data = {}
        data['canonical'] = self.canonical.detach().cpu().numpy()
        data['exp_face'] = self.exp_face.detach().cpu().numpy()
        
        np.savez(save_file, **data)
        
        return
    
    def loadData(self, load_file = 'BlendModel.npz'):
        data = np.load(load_file)
        
        self.canonical = nn.Parameter(to_tensor(data['canonical']), requires_grad=True)
        self.exp_face = nn.Parameter(to_tensor(data['exp_face']), requires_grad=True)
        
        return 