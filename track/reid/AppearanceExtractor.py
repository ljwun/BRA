import os.path as osp
import sys
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.append(__proot__)
sys.path.append(osp.join(__proot__,  "third_party", "fast-reid"))

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer

class AppearanceExtractor:
    def __init__(self, device):
        cfg_file = osp.join(__proot__,  "third_party", "fast-reid", "configs", "MSMT17", "sbs_S50.yml")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_file)
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = osp.join(__proot__,  "pretrain", "msmt_sbs_S50.pth")
        self.cfg.freeze()

        self.extractor = build_model(self.cfg)
        self.extractor.eval()
        Checkpointer(self.extractor).load(self.cfg.MODEL.WEIGHTS)

    def extract(self, imgs):
        feats = []
        for img in imgs:
            if img is None:
                feats.append(np.ones(2048))
                continue
            inTensor={"images": self.__pre_process(img)}
            with torch.no_grad():
                outTensor = self.extractor(inTensor)
            feats.append(self.__post_process(outTensor))
        return np.vstack(feats)

    def extract_with_crop(self, frame, dets, filterSize=None):
        imgs = []
        if dets.shape[1] == 5:
            scores = dets[:, 4]
        else:
            dets = dets.cpu().numpy()
            scores = dets[:, 4] * dets[:, 5]
        bboxes = dets[:, :4]
        for (x1, y1, x2, y2), score in zip(bboxes, scores):
            x1, y1 = max(0, round(x1)), max(0, round(y1))
            x2, y2 = min(frame.shape[1], round(x2)), min(frame.shape[0], round(y2))
            if (filterSize is not None and score < filterSize):
                imgs.append(None)
                continue
            imgs.append(frame[round(y1):round(y2), round(x1):round(x2)])
        return self.extract(imgs)

    def __pre_process(self, img):
        inTensor = img[:, :, ::-1]
        try:
            inTensor = cv2.resize(
                inTensor, 
                tuple(self.cfg.INPUT.SIZE_TEST[::-1]),
                interpolation=cv2.INTER_CUBIC
            )
        except Exception as e:
            print(f'inTensor size = {inTensor.shape}')
            raise e
        inTensor = torch.as_tensor(
            inTensor.astype("float32").transpose(2, 0, 1)
        )[None]
        if self.cfg.MODEL.DEVICE == 'cpu':
            return inTensor
        return inTensor.to(self.cfg.MODEL.DEVICE)

    def __post_process(self, inTensor):
        feat = F.normalize(inTensor)
        if self.cfg.MODEL.DEVICE == 'cpu':
            return feat.data.numpy()
        return feat.cpu().data.numpy()