import cv2
import numpy as np

def VISBlockText(
    frame,
    text_list,
    start_position,
    ratio, thickness,
    fg_color, bg_color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    point_reverse=(False, False)
):
    txt_sizes = np.asarray(
            [
                cv2.getTextSize(
                    text, font,
                    2*ratio, 2*ratio)[0] 
                for text in text_list
            ]
        )
    x_size = txt_sizes[:, 0].max()+1
    y_size = int(1.5*txt_sizes[:, 1].sum())
    x_start = start_position[0]
    x_end = start_position[0] + x_size
    if point_reverse[0]:
        x_start = frame.shape[1] - x_end
        x_end = frame.shape[1] - start_position[0]
    y_start = start_position[1]
    y_end = start_position[1] + y_size
    if point_reverse[1]:
        y_start = frame.shape[0] - y_end
        y_end = frame.shape[0] - start_position[1]

    if bg_color is not None:
        crop = frame[y_start:y_end, x_start:x_end]
        patch = np.ones_like(crop) * np.array(bg_color[:3])
        alpha = 1.0
        if len(bg_color)==4:
            alpha = bg_color[3]
        patch = (crop*(1-alpha) + patch*alpha).astype(np.uint8)
        frame[y_start:y_end, x_start:x_end] = patch

    for i, text in enumerate(text_list):
        cv2.putText(
            frame, text,
            (
                x_start, 
                y_start + int(i *1.5*txt_sizes[:, 1].max() + 1.25*txt_sizes[i][1])
            ),
            font, 2*ratio, fg_color[:3], thickness=thickness
        )
    return (
            x_end if not point_reverse[0] else frame.shape[1] - x_start,
            y_end if not point_reverse[1] else frame.shape[0] - y_start
        )