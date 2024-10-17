import os
from abc import ABC
from glob import glob
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from face_detector import FaceDetector
from image import crop_image_bbox, squarefiy, get_bbox

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from IPython import embed

from faceParsing.FaceParse import FaceParse

MP_TASK_FILE='face_landmarker.task'

class FaceMeshDetector:
    def __init__(self, MP_TASK_FILE = "face_landmarker.task"):
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1)
        self.model = mp_python.vision.FaceLandmarker.create_from_options(
            options)
 

    def getBlend(self, blendshapes):
        res = np.zeros(52)
        for i, bld in enumerate(blendshapes):
            res[i] = bld.score
        return res
    
    def getLmk(self, lmks):
        res = np.zeros((478, 3))
        for i, lmk in enumerate(lmks):
            res[i][0], res[i][1], res[i][2] = lmk.x, lmk.y, lmk.z
        return res
    
    def update(self, frame):
        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.model.detect(frame_mp)
        blend = self.getBlend(result.face_blendshapes[0])
        lmk = self.getLmk(result.face_landmarks[0])
        matrix = result.facial_transformation_matrixes[0]

        return lmk, blend, matrix


class GeneratorDataset(Dataset, ABC):
    def __init__(self, source, config):
        self.device = 'cuda:0'
        self.config = config
        self.source = Path(source)

        self.initialize()
        self.face_detector_mediapipe = FaceDetector('google')
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)
        
        self.faceDetctor = FaceMeshDetector(MP_TASK_FILE)
        
        self.faceparse = FaceParse(pretrained_model = './faceParsing/79999_iter.pth')
        

    def initialize(self):
        path = Path(self.source, 'source')
        if not path.exists() or len(os.listdir(str(path))) == 0:
            video_file = self.source / 'video.mp4'
            if not os.path.exists(video_file):
                logger.error(f'[ImagesDataset] Neither images nor a video was provided! Execution has stopped! {self.source}')
                exit(1)
            path.mkdir(parents=True, exist_ok=True)
            os.system(f'ffmpeg -i {video_file} -vf fps={self.config.fps} -q:v 1 -start_number 0 {self.source}/source/%05d.png')

        self.images = sorted(glob(f'{self.source}/source/*.jpg') + glob(f'{self.source}/source/*.png'))

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        dense_lmks = self.face_detector_mediapipe.dense(image)
        return lmks, dense_lmks

    def run(self):
        logger.info('Generating dataset...')
        bbox = None
        bbox_path = self.config.actor + "/bbox.pt"

        if os.path.exists(bbox_path):
            bbox = torch.load(bbox_path)

        dct = {}
        for imagepath in tqdm(self.images):
            lmk_path = imagepath.replace('source', 'kpt').replace('png', 'npy').replace('jpg', 'npy')
            lmk_path_dense = imagepath.replace('source', 'kpt_dense').replace('png', 'npy').replace('jpg', 'npy')
            blend_path = imagepath.replace('source', 'blend').replace('png', 'npz').replace('jpg', 'npz')
            
            image_path = imagepath.replace('source', 'images')
            mask_path = imagepath.replace('source', 'mask').replace('jpg', 'png')
            Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
            

            if not os.path.exists(lmk_path) or not os.path.exists(lmk_path_dense) or not os.path.exists(blend_path):    
                image = cv2.imread(imagepath)
                h, w, c = image.shape

                if bbox is None and self.config.crop_image:
                    lmk, _ = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # estimate initial bbox
                    bbox = get_bbox(image, lmk, bb_scale=self.config.bbox_scale)
                    torch.save(bbox, bbox_path)

                if self.config.crop_image:
                    image = crop_image_bbox(image, bbox)
                    if self.config.image_size[0] == self.config.image_size[1]:
                        image = squarefiy(image, size=self.config.image_size[0])
                else:
                    image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]), interpolation=cv2.INTER_CUBIC)

                lmk, dense_lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                source_land, source_blendshape, s_matrix =  self.faceDetctor.update(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                dct['lmk'] = source_land
                dct['blend'] = source_blendshape
                dct['matrix'] = s_matrix
                
                if lmk is None:
                    logger.info(f'Empty face_alignment lmks for path: ' + imagepath)
                    lmk = np.zeros([68, 2])

                if dense_lmk is None:
                    logger.info(f'Empty mediapipe lmks for path: ' + imagepath)
                    dense_lmk = np.zeros([478, 2])

                Path(lmk_path).parent.mkdir(parents=True, exist_ok=True)
                Path(lmk_path_dense).parent.mkdir(parents=True, exist_ok=True)
                Path(imagepath.replace('source', 'images')).parent.mkdir(parents=True, exist_ok=True)
                Path(blend_path).parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(imagepath.replace('source', 'images'), image)
                
                np.save(lmk_path_dense, dense_lmk)
                np.save(lmk_path, lmk)
    
                np.savez(blend_path, **dct)
            
            if not os.path.exists(mask_path):
                self.faceparse.solveImageFile(image_path = image_path, mask_path = mask_path)
                
                
            # print("In generating dataset!")
            # from IPython import embed