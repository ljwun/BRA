import yaml
import numpy as np
import cv2

class Mapper:
    def __init__(self, configPath, biasX=0, biasY=0):
        # loading yaml config file
        configStream = open(configPath, 'r')
        config = yaml.safe_load(configStream)
        configStream.close()
        # loading mapping parameter
        points = config['BEV']['position']
        points = [(p['x'], p['y']) for p in points]
        points = np.asarray(points, dtype='float32')
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