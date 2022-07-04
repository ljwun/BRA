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

from compute_block import ACBlock
from mask.checker import MaskChecker
from detect import Detector
from tracker.byte_tracker import BYTETracker
from distance.mapping import Mapper
from scipy.spatial.distance import cdist
from wash_hand.eventFilter import WashingFilter

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
        "track_buffer":30,
        "match_thresh":0.8,
        "aspect_ratio_thresh":1.6,
        "min_box_area":10.0,
        "mot20":False,
    }

    mask_detector = MaskChecker("cuda:0", 'm', 'New_FMD_m1k.pth')
    person_detector = Detector('m', 0, True, True)
    byteTracker = BYTETracker(type('',(object,),para)(), frame_rate=fps)
    mapper = Mapper(confPath)
    washFilter = WashingFilter(confPath)
    record = pd.DataFrame(columns=['type', 'deathCounter'])

    mask_metrics = ACBlock(fps*60*10, fps*5)
    distance_metrics = ACBlock(fps*60*10, fps*5)
    hand_wash_metrics = ACBlock(fps*60*10, fps*5)
    wash_correct_metrics = ACBlock(fps*60*10, fps*5)

    frame_id = 0
    frame_bound = fps * 60 * 12
    life = 50
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
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
            notWashIds, wrongWashIds, correctWashIds = washFilter.work(online_persons)
            # [wash-hand visualization]
            record.iloc[:]['deathCounter'] += 1
            record = record.loc[record['deathCounter'] < life]
            for i in correctWashIds: record.loc[i] = [1, 0]
            for i in wrongWashIds: record.loc[i] = [2, 0]
            for i in notWashIds: record.loc[i] = [3, 0]
            for target in online_persons:
                tID = target.track_id
                if tID not in record.index:
                    continue
                tType = record.loc[tID, 'type']
                record.loc[tID, 'deathCounter'] = 0
                xmin, ymin, w, h = target.tlwh
                box = tuple(map(int, (xmin, ymin, xmin+w, ymin+h)))
                if tType == 1:
                    color = (255, 255, 255)
                    text = 'OK'
                elif tType == 2:
                    color = (176, 21, 61)
                    text = 'wrong'
                elif tType == 3:
                    color = (82, 4, 28)
                    text = 'bad'
                cv2.rectangle(frame, box[0:2], box[2:4], color=color, thickness=5)
                cv2.putText(frame, f'{text}|{tID}|{w/h:.3f}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 5, color, thickness=3)
            
            for target in online_persons:
                tID = target.track_id
                tracer = washFilter.record
                if tID not in tracer.index:
                    continue
                xmin, ymin, w, h = target.tlwh
                box = tuple(map(int, (xmin, ymin, xmin+w, ymin+h)))
                # 處理出廁所
                if np.isnan(tracer.loc[tID, 'cleanCounter']):
                    color = (0, 255, 255)
                    text = f"dirty={tracer.loc[tID, 'dirtyCounter']}"
                else:
                    color = (255, 153, 255)
                    text = f"clean={tracer.loc[tID, 'cleanCounter']}"
                cv2.rectangle(frame, box[0:2], box[2:4], color=color, thickness=3)
                cv2.putText(frame, f'{tID}|{text}|{w/h:.3f}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=3)
            
            # [social distance and visualization]
            total_distance, total_edge_num = 0, 0
            if len(online_persons) > 0:
                bottom_center_points = np.asarray([(p.tlwh[0]+p.tlwh[2]/2, p.tlwh[1]+p.tlwh[3]) for p in online_persons])
                IPM_points = mapper.points_warp(bottom_center_points)
                distance = cdist(IPM_points, IPM_points, 'euclidean')
                # distance is a diagonal zero symmetric matrix
                # following formula is : sum(distance matrix) / 2 / C(points count, 2)
                # equal to : sum(distance matrix) / P(points count, 2)
                total_distance = distance.sum()
                total_edge_num = IPM_points.shape[0] * (IPM_points.shape[0]-1)
                
                # [distance edge visualization]
                danger_idx = undirectedFilter(distance, 0, 150.0, True)
                warning_idx = undirectedFilter(distance, 150.0, 250.0)
                for point in bottom_center_points:
                    cv2.circle(frame, (np.int32(point[0]), np.int32(point[1])), 6, (0, 255, 0), -1, 8, 0)
                for edge in warning_idx:
                    cv2.line(frame,
                            np.int32(bottom_center_points[edge[0]]),
                            np.int32(bottom_center_points[edge[1]]),
                            (0, 133, 242), 3)
                for edge in danger_idx:
                    cv2.line(frame,
                            np.int32(bottom_center_points[edge[0]]),
                            np.int32(bottom_center_points[edge[1]]),
                            (0, 0, 255), 6)

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
		f'Real-time:',
                f'Person Count: {len(online_persons):d}',
                f'Average Distance: {avg_distance / 100:.2f} m',
                f'Mask Check: {with_mask_ratio * 100:.2f}%',
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_sizes = np.asarray([cv2.getTextSize(text, font, 2, 2)[0] for text in texts])
            txt_color = (0 ,0 ,0)
            txt_bk_color = (255, 255, 255)
            cv2.rectangle(
                frame, (0, 0), 
                (txt_sizes[:, 0].max()+1, int(1.5*txt_sizes[:, 1].sum())),
                color=txt_bk_color, thickness=-1
            )
            for i, text in enumerate(texts):
                cv2.putText(
                    frame, text,
                    (0, int(i *1.5*txt_sizes[:, 1].max() + 1.25*txt_sizes[i][1])),
                    font, 2, txt_color, thickness=3
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
                f'Wash Hand After Toilet:',
                f'      Washed Ratio: {washed_avg * 100:.2f}%',
                f'      Correct Ratio: {wash_correct_avg * 100:.2f}%'
            ]
            PTA_txt_sizes = np.asarray([cv2.getTextSize(text, font, 2, 2)[0] for text in PTA_texts])
            PTA_xmin, PTA_ymin = 0, int(1.5*txt_sizes[:, 1].sum())+1
            PTA_w, PTA_h = PTA_txt_sizes[:, 0].max()+1, int(1.5*PTA_txt_sizes[:, 1].sum())+1
            cv2.rectangle(
                frame, 
                (PTA_xmin, PTA_ymin),
                (PTA_xmin+PTA_w, PTA_ymin+PTA_h),
                color=txt_color, thickness=-1
            )
            for i, text in enumerate(PTA_texts):
                cv2.putText(
                    frame, text,
                    (0, PTA_ymin + int(i *1.5*PTA_txt_sizes[:, 1].max() + 1.25*PTA_txt_sizes[i][1])),
                    font, 2, txt_bk_color, thickness=3
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

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def undirectedFilter(distance, r0:float, r1:float, equal:bool=False):
    rd_half = (r1 - r0) / 2.0
    filtered = np.where(abs(distance-r0-rd_half) < rd_half) if not equal else np.where(abs(distance-r0-rd_half) <= rd_half)
    formated = []
    for i in range(filtered[0].shape[0]):
        if filtered[1][i] != filtered[0][i] and\
            (filtered[1][i], filtered[0][i]) not in formated:
            formated.append((filtered[0][i], filtered[1][i]))
    return formated 

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
    ]
).astype(np.float32).reshape(-1, 3)


if __name__ == "__main__":
    main(
        "212_20220331105927.avi",
        '212_conf.yaml',
        "[WORK2]212_20220331105927.mkv"
    )
