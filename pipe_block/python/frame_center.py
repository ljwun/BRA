import numpy as np
import cv2
from typing import Any, List, Tuple
from collections import deque
import itertools
import time

class FrameCenter:
    def __init__(self, video_path:str, max_batch:int=1, start_second:int=None, frame_step:int=1, fps:int=None)->None:
        self.read_gap = frame_step - 1
        self.max_batch = max_batch
        self.frame_bfr = deque()
        self.current_pipe = None
        self.base_frame_ID = 1
        self.video_path=video_path
        self.setting_fps = fps
        self.start_second = start_second
        self.on = False
    
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

    def Allocate(self)->Tuple[deque[np.ndarray], int, deque[int], bool]:
        if len(self.frame_bfr) == 0:
            return deque(), 0, deque(), True
        if len(self.frame_bfr[0]) > self.max_batch:
            bfr = deque(itertools.islice(self.frame_bfr[0], self.max_batch))
            self.frame_bfr[0] = deque(itertools.islice(self.frame_bfr[0], self.max_batch, None))
            fids = deque(range(self.base_frame_ID, self.base_frame_ID+self.max_batch))
            self.base_frame_ID = self.base_frame_ID + self.max_batch
            return bfr, self.max_batch, fids, False
        bfr = self.frame_bfr[0].copy()
        self.frame_bfr[0].clear()
        fids = deque(range(self.base_frame_ID, self.base_frame_ID+len(bfr)))
        self.base_frame_ID = self.base_frame_ID + len(bfr)
        if len(self.frame_bfr) > 1 or self.current_pipe is None:
            self.frame_bfr.popleft()
            self.base_frame_ID = 1
            return bfr, len(bfr), fids, True
        return bfr, len(bfr), fids, False

    def Load(self)->None:
        for _ in range(self.max_batch):
            if self.current_pipe is None:
                self.init_capture()
            ret, frame = self.cap.read()
            if not ret:
                self.current_pipe = None
                break
            if self.current_pipe is None:
                self.current_pipe = deque()
                self.frame_bfr.append(self.current_pipe)
            self.current_pipe.append(frame)
            for _ in range(self.read_gap):
                ret, _ = self.cap.read() if ret else (False, None)
            if not ret:
                self.current_pipe = None
                break

    def Async_Load(self)->None:
        self.init_capture()
        while self.on:
            ret, frame = self.cap.read()
            if not ret:
                self.current_pipe = None
                time.sleep(1)
                self.init_capture()
            else:
                if self.current_pipe is None:
                    self.current_pipe = deque()
                    self.frame_bfr.append(self.current_pipe)
                self.current_pipe.append(frame)
                for _ in range(self.read_gap):
                    ret, _ = self.cap.read() if ret else (False, None)
    
    def Get(self, meta_type:int)->Any:
        return self.cap.get(meta_type)

    def Exit(self)->None:
        self.cap.release()
