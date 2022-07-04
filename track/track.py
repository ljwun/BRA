import sys
import cv2
import os
import os.path as osp

__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking

def video_demo(args, detector, vid_path):
    cap = cv2.VideoCapture(vid_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = osp.join(__proot__, "test", "vid_result")
    os.makedirs(save_folder, exist_ok=True)
    vid_writer = cv2.VideoWriter(
        osp.join(save_folder, "result.mkv"),
        cv2.VideoWriter_fourcc(*'X264'),
        fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    frame_id = 0
    results = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id > fps * 60:
                break
            outputs, img_info = detector.detect(frame)
            if outputs[0] is None:
                vid_writer.write(img_info['raw'])
                continue
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], detector.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh= t.tlwh       #tuple represented (x, y ,w ,h)
                tid = t.track_id
                if(
                    tlwh[2] * tlwh[3] < args.min_box_area or
                    tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                ):continue
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
            online_im = plot_tracking(
                img_info['raw'], online_tlwhs, online_ids, frame_id, fps=30
            )
            vid_writer.write(online_im)
    except KeyboardInterrupt:
            print("Interupt...Shutdown...ok...")

    vid_writer.release()
    cap.release()
    res_file = osp.join(save_folder, "result.txt")
    with open(res_file, 'w') as f:
        f.writelines(results)


