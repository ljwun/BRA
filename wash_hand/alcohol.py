import pandas as pd
import yaml
from numpy import isnan


class TargetFilter:
    def __init__(self, configPath):
        configStream = open(configPath, 'r')
        self.config = yaml.safe_load(configStream)
        configStream.close()
        self.record = pd.DataFrame(columns=['targetCounter', 'nonTargetCounter'])
        # ================= for out =====================================
        self.outdoorTrigger = self.__virtualFence((0.5, 1), self.config['Fences']['outdoor'])
        self.outdoorIOTrigger = self.__edgeTrigger()
        # ================= for in ======================================\
        self.alcoholLeftTrigger = self.__virtualFence((0,1), self.config['Fences']['alcohol_LEFT'])
        self.alcoholRightTrigger = self.__virtualFence((1,1), self.config['Fences']['alcohol_RIGHT'])
        self.checkActionTrigger = self.__checkAction((self.config['Threshold']['aspect-ratio']))
        # ================= Threshold ===================================
        self.targetThreshold = self.config['Threshold']['clean']
        self.nonTargetThreshold = self.config['Threshold']['dirty']
        self.waitingThreshold = self.config['Threshold']['waiting']

    def work(self, inputs):
        # inputs are list of object data with its properties
        # all Trigger Functions define below would filtrate input
        # and return custom result
        outdoorTable = self.outdoorTrigger(inputs)
        outdoorIOTable = self.outdoorIOTrigger(inputs, outdoorTable)
        alcoholLeftTable = self.alcoholLeftTrigger(inputs)
        alcoholLeftTable = self.checkActionTrigger(inputs, alcoholLeftTable)
        alcoholRightTable = self.alcoholRightTrigger(inputs)
        alcoholRightTable = self.checkActionTrigger(inputs, alcoholRightTable)

        # Step 0: 迭代這一輪的nonTargetCounter
        self.record.iloc[:]['nonTargetCounter'] += 1

        # Step 1: 對目標對象的record更新Counter
        # 考慮到計算雜訊，當再次進行目標行為時，補足目標判定時間
        # 能減少斷斷續續的目標行為偵測結果
        # 只有持續不進行目標行為（累積nonTargetCounter）才視為離開狀態了
        for i, obj in enumerate(inputs):
            # pid 作為record的index，直接索引
            pid = obj.track_id
            if (
                (alcoholRightTable[i] or alcoholLeftTable[i]) and 
                pid in self.record.index
            ):
                if isnan(self.record.loc[pid, 'targetCounter']):
                    self.record.loc[pid, 'targetCounter'] = 0
                    self.record.loc[pid, 'nonTargetCounter'] = 0
                else:
                    self.record.loc[pid, 'targetCounter'] += self.record.loc[pid, 'nonTargetCounter']
                    self.record.loc[pid, 'nonTargetCounter'] = 0

        # Step 2: 維護離開室外對象的record
        # 主要有兩部分:
        # A: 進入室內（狀態變換，由IOTable定義），記錄進record
        # B: 進入室內後仍在室外檢查區域者，不被視為離開，則record的nonTargetCounter會保持0
        for i, obj in enumerate(inputs):
            pid = obj.track_id
            if outdoorIOTable[i]:
                self.record.loc[pid] = [None, 0]

        # Step 3: 最終檢查record的步驟
        # 會以Counter來過濾出
        #   (1). 超時不進行行為的人 => 視為離開了
        #   (2). 行為時間不滿足的人 => 視為不確實進行行為
        #   (3). 行為時間滿足的人 => 視為已進行行為了
        if self.record.size == 0:
            return [], [], []

        nonTarget = isnan(self.record['targetCounter']) & (self.record['nonTargetCounter'] >= self.nonTargetThreshold)
        nonTargetIds = self.record.loc[nonTarget].index
        self.record = self.record.loc[~nonTarget]

        wrongTarget = ~isnan(self.record['targetCounter']) & (self.record['nonTargetCounter'] >= self.waitingThreshold)
        wrongTargetIds = self.record.loc[wrongTarget].index
        self.record = self.record.loc[~wrongTarget]

        correctTarget = self.record['targetCounter'] >= self.targetThreshold
        correctTargetIds = self.record.loc[correctTarget].index
        self.record = self.record.loc[~correctTarget]
        return nonTargetIds, wrongTargetIds, correctTargetIds

    def __virtualFence(self, vertex_type, fence, fn=None):
        # we apply Point in Polygon Strategies
        # to check whether a single vertex is in specified range or not
        # vertex_type = (0,0)->left top
        #               (0,1)->left bottom
        #               (1,1)->right bottom
        #               (1,0)->right top
        def instanceFence(objs):
            result = []
            for obj in objs:
                p = {
                    'x':obj.tlwh[0]+vertex_type[0]*obj.tlwh[2], 
                    'y':obj.tlwh[1]+vertex_type[1]*obj.tlwh[3]
                }
                result.append(windingNumber(p, fence) != 0)
            return result
        return instanceFence

    def __edgeTrigger(self):
        pre_filtered_ids = []
        def instanceEdgeTrigger(objs, filtered_result):
            result = []
            current_filtered_ids = []
            nonlocal pre_filtered_ids
            for i, tag in enumerate(filtered_result):
                if not tag:
                    result.append(objs[i].track_id in pre_filtered_ids)
                else:
                    current_filtered_ids.append(objs[i].track_id)
                    result.append(False)
            pre_filtered_ids = current_filtered_ids
            return result
        return instanceEdgeTrigger

    def __checkAction(self, thres, fn=None):
        def instanceCheck(objs, filtered_result):
            result = []
            for i, tag in enumerate(filtered_result):
                if tag:
                    aspect_ratio = objs[i].tlwh[2] / objs[i].tlwh[3]
                    result.append(thres[0] < aspect_ratio < thres[1])
                else:
                    result.append(False)
            return result
        return instanceCheck


def windingNumber(point, vertexs):
    # https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html
    def isLeft(p0, p1, p2):
        return ( (p1['x']-p0['x']) * (p2['y']-p0['y']) - (p2['x']-p0['x'])  *   (p1['y']-p0['y']) )
    wn = 0
    for i in range(len(vertexs)):
        if vertexs[i]['y'] <= point['y']:
            if vertexs[(i+1)%len(vertexs)]['y'] > point['y']:
                if isLeft(vertexs[i], vertexs[(i+1)%len(vertexs)], point) > 0:
                    wn += 1
        else:
            if vertexs[(i+1)%len(vertexs)]['y'] <= point['y']:
                if isLeft(vertexs[i], vertexs[(i+1)%len(vertexs)], point) < 0:
                    wn -= 1
    return wn