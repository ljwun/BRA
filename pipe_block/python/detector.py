from loguru import logger
import os, re, time
import os.path as osp
import torch
import numpy as np
import cv2

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), "..", ".."))
from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, postprocess
from yolox.exp import get_exp

class Detector:
    def __init__(self, device, exp_path, checkpoint, fuse, fp16, cls_name, legacy=False):
        self.device = re.match("cpu|cuda(:\d*)?", device)
        assert self.device != None, f"Cannot resolve target device string ${device}"
        self.device = self.device.group(0)
        dinfo = self.device.split(":")
        if dinfo[0] == "cuda":
            assert(
                int(dinfo[1]) <= torch.cuda.device_count()
            ), "Cannot find target device"
        
        self.exp = get_exp(exp_path)
        self.model = self.exp.get_model().to(self.device)
        self.model.eval()
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt["model"])
        self.preproc = ValTransform(legacy=legacy)

        if fuse:
            self.model = fuse_model(self.model)
        if fp16:
            self.model = self.model.half()
        
        self.fuse = fuse
        self.fp16 = fp16

        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre

        self.test_size = self.exp.test_size
        self.cls_names = cls_name
        self._COLORS = _COLORS

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

        imgs = np.stack([self.preproc(img, None, self.test_size)[0] for img in imgs])
        imgs = torch.from_numpy(imgs).float().to(self.device)
        if self.fp16:
            imgs = imgs.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(imgs)
            logger.trace("Infer time: {:.4f}s".format(time.time() - t0))
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, imgs_info
    
    def visual_rp(self, frame, outputs, info, cls_conf=0.5, color_map=None):
        color_map = self._COLORS if color_map is None else color_map
        for box, score, pred_class in zip(
            outputs[:, 0:4]/info["ratio"], 
            outputs[:, 4] * outputs[:, 5],
            outputs[:, 6]
        ):
            if score < cls_conf:
                continue
            # property
            pred_class = int(pred_class)
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # [visualization]
            color = (color_map[pred_class] * 255).astype(np.uint8).tolist()
            text = f'{self.cls_names[pred_class]}:{score*100:.1f}'
            txt_color = (0, 0, 0) if np.mean(color_map[pred_class]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.7, 1)[0]
            txt_bk_color = (color_map[pred_class] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=3)
            cv2.rectangle(
                            frame, (xmin, ymin+1), 
                            (xmin+txt_size[0]+1, ymin + int(1.5*txt_size[1])),
                            color=txt_bk_color, thickness=-1
            )
            cv2.putText(frame, text, (xmin, ymin+txt_size[1]), font, 0.7, txt_color, thickness=1)
        return
    
    def output_tidy(self, outputs, info, cls_conf=0.5):
        format_out = {}
        for clsE in self.cls_names:
            format_out[clsE] = []
        for box, score, pred_class in zip(
            outputs[:, 0:4]/info["ratio"], 
            outputs[:, 4] * outputs[:, 5],
            outputs[:, 6]
        ):
            if score < cls_conf:
                continue
            # property
            pred_class = int(pred_class)
            # main work
            format_out[self.cls_names[pred_class]].append({
                'bbox':box,
                'score':score,
            })
        return format_out

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)