import os.path as osp
import sys
import cv2
import numpy as np
from loguru import logger

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)
sys.path.append(osp.join(__proot__,  "third_party", "YOLOX"))
sys.path.append(osp.join(__proot__, "third_party", "ByteTrack", "yolox"))

import compute_block as cmb
from mask.checker import MaskChecker
from detect import Detector
from tracker.byte_tracker import BYTETracker
from track.byte_tracker_reid import BYTETracker_reid
from track.reid import AppearanceExtractor
from distance.mapping import Mapper
from distance.visual import WarningLine
from scipy.spatial.distance import cdist
from wash_hand import EventFilter

from .BaseWorker import BaseWorker

class Worker(BaseWorker):
    def __init__(
        self,
        vin_path, 
        conf_path, track_parameter,
        record_life=2,
        metrics_duration=600,
        metrics_update_time=5,
        actual_framerate=None,
        reid=False,
        start_frame=None
    ):
        super().__init__()
        self.cap = cv2.VideoCapture(vin_path)
        if not self.cap.isOpened():
            raise Exception(f'Could not open file "{vin_path}"!')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_framerate is not None:
            self.fps = actual_framerate
        self.frameID = 0
        self.retval = True
        self.reid = reid
        if start_frame is not None:
            for _ in range(start_frame):
                self.cap.read()

        self.MDetector = MaskChecker("cuda:0", 'm', 'New_FMD_m1k.pth')
        self.PDetector = Detector('m', 0, True, True)
        if self.reid:
            self.byteTracker = BYTETracker_reid(type('',(object,),track_parameter)(), frame_rate=self.fps)
            self.appearanceExtractor = AppearanceExtractor("cuda:0")
        else:
            self.byteTracker = BYTETracker(type('',(object,),track_parameter)(), frame_rate=self.fps)
        self.mapper = Mapper(conf_path)
        self.alcoholFilter = EventFilter(conf_path, record_life, self.fps)

        self.mask_metrics = cmb.ACBlock(
            self.fps*metrics_duration,
            self.fps*metrics_update_time
        )
        self.distance_metrics = cmb.ACBlock(
            self.fps*metrics_duration,
            self.fps*metrics_update_time
        )
        self.hand_wash_metrics = cmb.ACBlock(
            self.fps*metrics_duration,
            self.fps*metrics_update_time
        )
        self.wash_correct_metrics = cmb.ACBlock(
            self.fps*metrics_duration,
            self.fps*metrics_update_time
        )

    def _workFlow(self):
        # [LEVEL_0_BLOCK]
        _, frame = self.cap.retrieve()

        # [LEVEL_1_BLOCK] === INPUT -> frame
        person_outputs, person_info = self.PDetector.detect(frame)
        mask_outputs, mask_info = self.MDetector.detect(frame)

        # [LEVEL_2_BLOCK] === INPUT -> LEVEL_1_PERSON_BLOCK result
        online_persons = []
        if person_outputs[0] is not None:
            if self.reid:
                feats = self.appearanceExtractor.extract_with_crop(frame, person_outputs[0], 0.1)
                online_persons = self.byteTracker.update(
                    person_outputs[0], 
                    [person_info[0]['height'], person_info[0]['width']], 
                    self.PDetector.test_size,
                    feats
                )
            else:
                online_persons = self.byteTracker.update(
                    person_outputs[0], 
                    [person_info[0]['height'], person_info[0]['width']], 
                    self.PDetector.test_size
                )
        with_mask, without_mask = [], []
        if mask_outputs[0] is not None:
            mask_out = mask_outputs[0].cpu()
            out = self.MDetector.output_tidy(mask_out, mask_info, 0.5)
            # mask_detector.visual_rp(frame, mask_out, mask_info, 0.5)
            with_mask = out['with_mask']+out['mask_wear_incorrect']
            without_mask = out['without_mask']
            
        # [LEVEL_3_BLOCK] === INPUT -> tracked ID and tracked person
        notWashIds, wrongWashIds, correctWashIds, c_table, b_table = self.alcoholFilter.work(online_persons)
        bottom_center_points = np.asarray([(p.tlwh[0]+p.tlwh[2]/2, p.tlwh[1]+p.tlwh[3]) for p in online_persons])
        IPM_points = self.mapper.points_warp(bottom_center_points)
        if bottom_center_points.shape[0] != 0:
            distance = cdist(IPM_points, IPM_points, 'euclidean')
        else:
            distance = np.zeros(0)
        
        # [LEVEL_4_BLOCK] === INPUT -> numerator and denominator
        total_distance = distance.sum() / 2
        total_edge_num = IPM_points.shape[0] * (IPM_points.shape[0]-1) / 2
        distance_period_avg = self.distance_metrics.step(total_distance, total_edge_num)
        mask_len = len(with_mask) + len(without_mask)
        mask_period_avg = self.mask_metrics.step(len(with_mask), mask_len)
        entry_len = len(notWashIds) + len(wrongWashIds) + len(correctWashIds)
        washed_len = len(wrongWashIds) + len(correctWashIds)
        washed_avg = self.hand_wash_metrics.step(washed_len, entry_len)
        wash_correct_avg = self.wash_correct_metrics.step(len(correctWashIds), washed_len)
  
        # [LEVEL_5_BLOCK] === INPUT -> metadata
        self.alcoholFilter.visualize(frame, online_persons)
        WarningLine(frame, bottom_center_points, distance, (0, 0, 255), 6, 0, 150.0, True)
        WarningLine(frame, bottom_center_points, distance, (0, 133, 242), 2, 150.0, 250.0)
        for point in bottom_center_points:
            cv2.circle(
                frame, 
                (np.int32(point[0]), np.int32(point[1])), 
                6, (0, 255, 0),
                -1, 8, 0
            )

        # [LEVEL_6_BLOCK] === INPUT -> assessment
        avg_distance = total_distance / total_edge_num if total_edge_num != 0 else 0
        with_mask_ratio = len(with_mask) / mask_len if mask_len != 0 else 1.0
        texts = [
            f'Real-time',
            f'Person Count: {len(online_persons):d}',
            f'Average Distance: {avg_distance / 100:.2f} m',
            f'Mask Check: {with_mask_ratio * 100:.2f}% ({mask_len})',
        ]
        cmb.VISBlockText(
            frame,
            texts, (0, 0),
            ratio=1, thickness=3,
            fg_color=(0 ,0 ,0), bg_color=(255, 255, 255),
            point_reverse=(False,True)
        )
        texts = [
            f'Period-time average:',
            f'Social Distance: {distance_period_avg / 100:.2f} m',
            f'Mask Worn Ratio: {mask_period_avg * 100:.2f}%',
            f'Disinfection After Entry:',
            f'      Washed Ratio: {washed_avg * 100:.2f}%',
            f'      Correct Ratio: {wash_correct_avg * 100:.2f}%'
        ]
        end_position = cmb.VISBlockText(
            frame,
            texts, (0, 0),
            ratio=1, thickness=3,
            fg_color=(255, 255, 255), bg_color=(0 ,0 ,0),
            point_reverse=(True,True)
        )

        # [LEVEL_7_BLOCK] === OUTPUT -> frame and assessment
        return self.frameID, frame

    def _conditionWork(self):
        self.frameID += 1
        self.retval = self.cap.grab()
        if not self.retval:
            return False
        return True

    def _endingWork(self):
        self.cap.release()
