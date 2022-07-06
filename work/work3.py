import os.path as osp
import sys
import cv2
import os
import numpy as np
import pandas as pd
from loguru import logger
import time

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(__proot__)
sys.path.append(osp.join(__proot__,  "third_party", "YOLOX"))
sys.path.append(osp.join(__proot__, "third_party", "ByteTrack", "yolox"))

import compute_block as cmb
from mask.checker import MaskChecker
from detect import Detector
from tracker.byte_tracker import BYTETracker
from distance.mapping import Mapper
from distance.visual import WarningLine
from scipy.spatial.distance import cdist
from wash_hand.alcohol import TargetFilter

def main(viname, confPath, voname):
    vid_path = osp.join(__proot__, "test", viname)
    cap = cv2.VideoCapture(vid_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = osp.join(__proot__, "test", "vid_result")
    os.makedirs(save_folder, exist_ok=True)
    vid_writer = cv2.VideoWriter(
        osp.join(save_folder, voname),
        cv2.VideoWriter_fourcc(*'X264'),
        fps, (int(width), int(height))
    )


    para = {
        "track_thresh":0.5,
        "track_buffer":125,
        "match_thresh":0.8,
        "aspect_ratio_thresh":1.6,
        "min_box_area":10.0,
        "mot20":False,
    }

    mask_detector = MaskChecker("cuda:0", 'm', 'New_FMD_m1k.pth')
    person_detector = Detector('m', 0, True, True)
    byteTracker = BYTETracker(type('',(object,),para)(), frame_rate=fps)
    mapper = Mapper(confPath)
    alcoholFilter = TargetFilter(confPath)
    record = pd.DataFrame(columns=['type', 'deathCounter'])

    mask_metrics = cmb.ACBlock(fps*60*10, fps*5)
    distance_metrics = cmb.ACBlock(fps*60*10, fps*5)
    hand_wash_metrics = cmb.ACBlock(fps*60*10, fps*5)
    wash_correct_metrics = cmb.ACBlock(fps*60*10, fps*5)

    frame_id = 0
    frame_bound = fps * 60 * 12
    life = 50
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            print(frame_id/25)
            if frame_id > frame_bound:
                break
            # [detection]
            person_outputs, person_info = person_detector.detect(frame)
            mask_outputs, mask_info = mask_detector.detect(frame)

            # [track]
            online_persons = []
            if person_outputs[0] is not None:
                t0 = time.time()
                online_persons = byteTracker.update(person_outputs[0], [person_info['height'], person_info['width']], person_detector.test_size)
                logger.info("Tracking time: {:.4f}s".format(time.time() - t0))
            # [wash-hand filter]
            notWashIds, wrongWashIds, correctWashIds = alcoholFilter.work(online_persons)
            # [wash-hand visualization]
            alcoholFilter.visualize(frame, online_persons)
            
            # [social distance and visualization]
            bottom_center_points = np.asarray([(p.tlwh[0]+p.tlwh[2]/2, p.tlwh[1]+p.tlwh[3]) for p in online_persons])
            for point in bottom_center_points:
                cv2.circle(frame, (np.int32(point[0]), np.int32(point[1])), 6, (0, 255, 0), -1, 8, 0)
            IPM_points = mapper.points_warp(bottom_center_points)
            distance = cdist(IPM_points, IPM_points, 'euclidean')
            total_distance = distance.sum() / 2
            total_edge_num = IPM_points.shape[0] * (IPM_points.shape[0]-1) / 2
            # [distance edge visualization]
            WarningLine(frame, bottom_center_points, distance, (0, 0, 255), 6, 0, 150.0, True)
            WarningLine(frame, bottom_center_points, distance, (0, 133, 242), 2, 150.0, 250.0)

            # [mask worm ratio and visualization]
            with_mask, without_mask = [], []
            if mask_outputs[0] is not None:
                mask_out = mask_outputs[0].cpu()
                out = mask_detector.output_tidy(mask_out, mask_info, 0.5)
                # mask_detector.visual_rp(frame, mask_out, mask_info, 0.5)
                with_mask = out['with_mask']+out['mask_wear_incorrect']
                without_mask = out['without_mask']

            # [real-time assessment visualization]
            avg_distance = total_distance / total_edge_num if total_edge_num != 0 else 0
            mask_len = len(with_mask) + len(without_mask)
            with_mask_ratio = len(with_mask) / mask_len if mask_len != 0 else 0
            texts = [
                f'Real-time',
                f'Person Count: {len(online_persons):d}',
                f'Average Distance: {avg_distance / 100:.2f} m',
                f'Mask Check: {with_mask_ratio * 100:.2f}%',
            ]
            end_position = cmb.VISBlockText(
                frame,
                texts, (0, 0),
                ratio=1, thickness=3,
                fg_color=(0 ,0 ,0), bg_color=(255, 255, 255)
            )

            # [period-time assessment visualization]
            restroom_len = len(notWashIds) + len(wrongWashIds) + len(correctWashIds)
            washed_len = len(wrongWashIds) + len(correctWashIds)
            distance_period_avg = distance_metrics.step(total_distance, total_edge_num)
            mask_period_avg = mask_metrics.step(len(with_mask), mask_len)
            washed_avg = hand_wash_metrics.step(washed_len, restroom_len)
            wash_correct_avg = wash_correct_metrics.step(len(correctWashIds), washed_len)
            PTA_texts = [
                f'Period-time average:',
                f'Social Distance: {distance_period_avg / 100:.2f} m',
                f'Mask Worn Ratio: {mask_period_avg * 100:.2f}%',
                f'Disinfection After Entry:',
                f'      Washed Ratio: {washed_avg * 100:.2f}%',
                f'      Correct Ratio: {wash_correct_avg * 100:.2f}%'
            ]
            end_position = cmb.VISBlockText(
                frame,
                PTA_texts, (0, 0),
                ratio=1, thickness=3,
                fg_color=(255, 255, 255), bg_color=(0 ,0 ,0),
                point_reverse=(True,False)
            )
            
            # [video]
            vid_writer.write(frame)
            
    except KeyboardInterrupt:
        print("Interupt...Shutdown...ok...")
    except:
        raise
    finally:
        vid_writer.release()
        cap.release()



if __name__ == "__main__":
    main(
        "217_20220522105956.avi",
        '217_conf.yaml',
        "[WORK3]217_20220522105956.mkv"
    )