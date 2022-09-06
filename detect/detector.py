from loguru import logger
import sys, re, time
import os.path as osp
import torch

import numpy as np
import cv2

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
from yolox.exp import get_exp
# from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess

class Detector:
    def __init__(self, device, exp_path, checkpoint, fuse, fp16):
        self.device = re.match("cpu|cuda(:\d*)?", device)
        assert self.device != None, f"Cannot resolve target device string ${device}"
        self.device = self.device.group(0)
        dinfo = self.device.split(":")
        if dinfo[0] == "cuda":
            assert(
                int(dinfo[1]) <= torch.cuda.device_count()
            ), "Cannot find target device"
        
        exp = get_exp(exp_path , None)
        self.test_size = exp.test_size
        self.model = exp.get_model().to(self.device)
        self.model.eval()
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt["model"])
        if fuse:
            self.model = fuse_model(self.model)
        if fp16:
            self.model = self.model.half()
        
        self.fuse = fuse
        self.fp16 = fp16

        self.num_classes = exp.num_classes

        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size

        self.means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
    
    def detect(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs_info = [{
            "height":img.shape[0],
            "width":img.shape[1],
            "raw":img,
            "ratio":min(
                self.test_size[0] / img.shape[0], 
                self.test_size[1] / img.shape[1]
            )
        } for img in imgs]

        imgs = np.stack([self.__preproc(img) for img in imgs])
        imgs = torch.from_numpy(imgs).float().to(self.device)
        if self.fp16:
            imgs = imgs.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(imgs)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

        return outputs, imgs_info

    def __preproc(self, image, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((self.test_size[0], self.test_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(self.test_size) * 114.0
        img = np.array(image)
        r = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if self.means is not None:
            padded_img -= self.means
        if self.std is not None:
            padded_img /= self.std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img
