import cv2
import numpy as np

def WarningLine(
    frame, 
    points, distance, 
    color, thickness,
    floor, ceiling,
    equal=False
):
    if len(points) == 0:
        return
    if ceiling < floor:
        floor, ceiling = ceiling, floor
    
    middle = (ceiling - floor) / 2.0
    partial = abs(distance - floor - middle)
    if not equal:
        filtered = np.where(partial < middle)
    else:
        filtered = np.where(partial <= middle)
    edges = []
    for i in range(filtered[0].shape[0]):
        if(
            filtered[1][i] != filtered[0][i] and
            (filtered[1][i], filtered[0][i]) not in edges
        ):
            edges.append((filtered[0][i], filtered[1][i]))
    
    for edge in edges:
        cv2.line(frame,
            np.int32(points[edge[0]]),
            np.int32(points[edge[1]]),
            color,
            thickness
        )
    