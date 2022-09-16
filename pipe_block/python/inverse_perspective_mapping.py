import yaml
import numpy as np
import cv2
from scipy.spatial.distance import cdist

class IPMer:
    def __init__(self, configPath):
        # loading yaml config file
        configStream = open(configPath, 'r')
        config = yaml.safe_load(configStream)
        configStream.close()
        # loading mapping parameter
        points = config['BEV']['position']
        points = [(p['x'], p['y']) for p in points]
        points = np.asarray(points, dtype='float32')
        biasX = config['BEV']['mapping']['biasX']
        biasY = config['BEV']['mapping']['biasY']
        mapW = config['BEV']['mapping']['width']
        mapH = config['BEV']['mapping']['height']
        dstPtr = np.float32([
            [biasX, biasY], [biasX, biasY+mapH],
            [biasX+mapW, biasY+mapH], [biasX+mapW, biasY]]
        )
        # calculate warp matrix
        self.matrix = cv2.getPerspectiveTransform(points,dstPtr)

    def point_warp(self, x, y):
        M = self.matrix
        d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
        return (
            int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),
            int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d)
        )

    def points_warp(self, points):
        M = self.matrix
        IPM_point = np.zeros(points.shape)
        for i in range(IPM_point.shape[0]):
            x, y = points[i][0], points[i][1]
            d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
            IPM_point[i, 0] = int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d)
            IPM_point[i, 1] = int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d)
        return IPM_point
    
    def calc_bev_distance(self, points):
        if len(points) == 0:
            return np.zeros(0)
        ipm_points = self.points_warp(points)
        return cdist(ipm_points, ipm_points, 'euclidean')

    @staticmethod
    def draw_warning_line(
        frame,
        points, distance, 
        floor=None, ceil=None, equal=False,
        color=(0,0,0), thickness=3
    ):
        if len(points) == 0:
            return
        ceil_table = np.full_like(distance, True, dtype=bool)
        floor_table = np.full_like(distance, True, dtype=bool)
        if ceil is not None:
            ceil_table = distance <= ceil if equal else distance < ceil
        if floor is not None:
            floor_table = distance >= floor if equal else distance > floor
        filtered = np.where(ceil_table & floor_table)
        edges = []
        for i in range(filtered[0].shape[0]):
            if filtered[1][i] > filtered[0][i]:
                edges.append((filtered[0][i], filtered[1][i]))
        
        for edge in edges:
            cv2.line(frame,
                np.int32(points[edge[0]]),
                np.int32(points[edge[1]]),
                color,
                thickness
            )