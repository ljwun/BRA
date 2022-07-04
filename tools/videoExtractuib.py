import cv2
import os
import os.path as osp
import sys
__proot__ = osp.normpath(osp.join(osp.dirname(__file__), ".."))

orin_file = osp.join(__proot__, "test", "20220331105927.avi")
cap = cv2.VideoCapture(orin_file)
save_folder = osp.join(__proot__, "test", "FaceMaskDet")
os.makedirs(save_folder, exist_ok=True)

frame_id = 0
counter, base = 900, 900
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter == base or counter < 0:
            counter = base
            cv2.imwrite(osp.join(save_folder, f"{frame_id}.png"), frame)
            frame_id += 1
            print(f"\r[{frame_id}]", end='')
        
        counter -= 1
except KeyboardInterrupt:
        print("\nInterupt...Shutdown...ok...")

cap.release()
print("\nFinish")