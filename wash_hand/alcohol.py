import pandas as pd
import yaml
from numpy import isnan
import cv2

class TargetFilter:
    def __init__(self, configPath, record_life=50):
        configStream = open(configPath, 'r')
        self.config = yaml.safe_load(configStream)
        configStream.close()
        self.sta_record = pd.DataFrame(columns=['targetCounter', 'nonTargetCounter'])
        self.dyn_record = pd.DataFrame(columns=['type', 'deathCounter'])
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
        self.dyn_life = record_life

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

        # Step 0: 迭代這一輪的nonTargetCounter和deathCounter
        self.sta_record.iloc[:]['nonTargetCounter'] += 1
        self.dyn_record.iloc[:]['deathCounter'] += 1
        self.dyn_record = self.dyn_record.loc[self.dyn_record['deathCounter'] < self.dyn_life]

        # Step 1: 對目標對象的record更新Counter
        # 考慮到計算雜訊，當再次進行目標行為時，補足目標判定時間
        # 能減少斷斷續續的目標行為偵測結果
        # 只有持續不進行目標行為（累積nonTargetCounter）才視為離開狀態了
        for i, obj in enumerate(inputs):
            # pid 作為record的index，直接索引
            pid = obj.track_id
            if (
                (alcoholRightTable[i] or alcoholLeftTable[i]) and 
                pid in self.sta_record.index
            ):
                if isnan(self.sta_record.loc[pid, 'targetCounter']):
                    self.sta_record.loc[pid, 'targetCounter'] = 0
                    self.sta_record.loc[pid, 'nonTargetCounter'] = 0
                else:
                    self.sta_record.loc[pid, 'targetCounter'] += self.sta_record.loc[pid, 'nonTargetCounter']
                    self.sta_record.loc[pid, 'nonTargetCounter'] = 0
            if pid in self.dyn_record.index:
                self.dyn_record.loc[pid, 'deathCounter'] = 0

        # Step 2: 維護離開室外對象的record
        # 主要有兩部分:
        # A: 進入室內（狀態變換，由IOTable定義），記錄進record
        # B: 進入室內後仍在室外檢查區域者，不被視為離開，則record的nonTargetCounter會保持0
        for i, obj in enumerate(inputs):
            pid = obj.track_id
            if outdoorIOTable[i]:
                self.sta_record.loc[pid] = [None, 0]

        # Step 3: 最終檢查record的步驟
        # 會以Counter來過濾出
        #   (1). 超時不進行行為的人 => 視為離開了
        #   (2). 行為時間不滿足的人 => 視為不確實進行行為
        #   (3). 行為時間滿足的人 => 視為已進行行為了
        # 並將過濾出的對象保存起來
        if self.sta_record.size == 0:
            return [], [], []

        nonTarget = isnan(self.sta_record['targetCounter']) & (self.sta_record['nonTargetCounter'] >= self.nonTargetThreshold)
        nonTargetIds = self.sta_record.loc[nonTarget].index
        for i in nonTargetIds: self.dyn_record.loc[i] = [1, 0]
        self.sta_record = self.sta_record.loc[~nonTarget]

        wrongTarget = ~isnan(self.sta_record['targetCounter']) & (self.sta_record['nonTargetCounter'] >= self.waitingThreshold)
        wrongTargetIds = self.sta_record.loc[wrongTarget].index
        for i in wrongTargetIds: self.dyn_record.loc[i] = [2, 0]
        self.sta_record = self.sta_record.loc[~wrongTarget]

        correctTarget = self.sta_record['targetCounter'] >= self.targetThreshold
        correctTargetIds = self.sta_record.loc[correctTarget].index
        for i in correctTargetIds: self.dyn_record.loc[i] = [3, 0]
        self.sta_record = self.sta_record.loc[~correctTarget]

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

    def visualize(self, frame, online_persons):
        for target in online_persons:
            tID = target.track_id
            if tID not in self.dyn_record.index:
                continue
            tType = self.dyn_record.loc[tID, 'type']
            xmin, ymin, w, h = target.tlwh
            box = tuple(map(int, (xmin, ymin, xmin+w, ymin+h)))
            if tType == 1:
                color = (82, 4, 28)
                text = 'bad'
            elif tType == 2:
                color = (176, 21, 61)
                text = 'wrong'
            elif tType == 3:
                color = (255, 255, 255)
                text = 'OK'
            cv2.rectangle(frame, box[0:2], box[2:4], color=color, thickness=5)
            cv2.putText(frame, f'{text}|{tID}|{w/h:.3f}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 5, color, thickness=3)

        for target in online_persons:
            tID = target.track_id
            if tID not in self.sta_record.index:
                continue
            xmin, ymin, w, h = target.tlwh
            box = tuple(map(int, (xmin, ymin, xmin+w, ymin+h)))
            # 處理出廁所
            if isnan(self.sta_record.loc[tID, 'targetCounter']):
                color = (0, 255, 255)
                text = f"dirty={self.sta_record.loc[tID, 'nonTargetCounter']}"
            else:
                color = (255, 153, 255)
                text = f"clean={self.sta_record.loc[tID, 'targetCounter']}"
            cv2.rectangle(frame, box[0:2], box[2:4], color=color, thickness=3)
            cv2.putText(frame, f'{tID}|{text}|{w/h:.3f}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=3)

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