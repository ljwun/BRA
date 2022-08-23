import numpy as np
import cv2
from typing import Any

class FrameCenter:
    def __init__(self, video_path:str, max_batch:int=1, start_frame:int=None)->None:
        self.End = False
        self.Finished = False
        self.max_batch = max_batch
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f'Could not open file "{video_path}"!')
        self.frame_bfr = []
        if start_frame is not None:
            for _ in range(start_frame):
                self.cap.read()
        self.base_frame_ID = 1
        self.Metadata = {
            'fps':self.cap.get(cv2.CAP_PROP_FPS),
            'height':self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'width':self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        }
    
    def Allocate(self)->tuple[list[np.ndarray], int, list[int]]:
        bfr = []
        if len(self.frame_bfr) >= self.max_batch:
            bfr = self.frame_bfr[:self.max_batch]
            self.frame_bfr = self.frame_bfr[self.max_batch:]
        else:
            bfr = self.frame_bfr[:]
            self.frame_bfr = []
        if len(self.frame_bfr) == 0 and self.End:
            self.Finished = True
        FIDs = [self.base_frame_ID + delta for delta in range(len(bfr))]
        self.base_frame_ID += len(bfr)
        return bfr, len(bfr), FIDs

    def Load(self)->None:
        for i in range(self.max_batch):
            ret, frame = self.cap.read()
            if not ret:
                self.End = True
                return
            self.frame_bfr.append(frame)
        return

    def Async_Load(self)->None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.End = True
                return
            self.frame_bfr.append(frame)
    
    def Get(self, meta_type:int)->Any:
        return self.cap.get(meta_type)

    def Exit(self)->None:
        self.cap.release()
