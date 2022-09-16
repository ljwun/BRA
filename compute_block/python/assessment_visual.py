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
        cv2.rectangle(
            frame, (x_start, y_start), 
            (x_end, y_end),
            color=bg_color, thickness=-1
        )

    for i, text in enumerate(text_list):
        cv2.putText(
            frame, text,
            (
                x_start, 
                y_start + int(i *1.5*txt_sizes[:, 1].max() + 1.25*txt_sizes[i][1])
            ),
            font, 2*ratio, fg_color, thickness=thickness
        )
    return (
            x_end if not point_reverse[0] else frame.shape[1] - x_start,
            y_end if not point_reverse[1] else frame.shape[0] - y_start
        )