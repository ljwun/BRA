import numpy as np
import cv2
from typing import Any, List, Tuple

class FrameCenter:
    def __init__(self, video_path:str, max_batch:int=1, start_second:int=None, frame_step:int=1, fps:int=None)->None:
        self.read_gap = frame_step - 1
        self.max_batch = max_batch
        self.frame_bfr = []
        self.base_frame_ID = 1
        self.video_path=video_path
        self.setting_fps = fps
        self.start_second = start_second
        self.init_capture()
    
    def init_capture(self)->None:
        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
           return
        self.Metadata = {
            'fps':int(self.cap.get(cv2.CAP_PROP_FPS)) if self.setting_fps is None else self.setting_fps,
            'height':self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'width':self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        }
        if self.start_second is not None:
            for _ in range(int(self.start_second*self.Metadata['fps'])):
                self.cap.read()
        self.base_frame_ID = 1

    def Allocate(self)->Tuple[List[np.ndarray], int, List[int]]:
        bfr = []
        if len(self.frame_bfr) >= self.max_batch:
            bfr = self.frame_bfr[:self.max_batch]
            self.frame_bfr = self.frame_bfr[self.max_batch:]
        else:
            bfr = self.frame_bfr[:]
            self.frame_bfr = []
        FIDs = [self.base_frame_ID + delta for delta in range(len(bfr))]
        self.base_frame_ID += len(bfr)
        return bfr, len(bfr), FIDs

    def Load(self)->None:
        for i in range(self.max_batch):
            ret, frame = self.cap.read()
            if ret:
                self.frame_bfr.append(frame)
            for _ in range(self.read_gap):
                ret, _ = self.cap.read() if ret else (False, None)
            if not ret:
                return
        return

    def Async_Load(self)->None:
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame_bfr.append(frame)
            for _ in range(self.read_gap):
                ret, _ = self.cap.read() if ret else (False, None)
            if not ret:
                return
    
    def Get(self, meta_type:int)->Any:
        return self.cap.get(meta_type)

    def Exit(self)->None:
        self.cap.release()
