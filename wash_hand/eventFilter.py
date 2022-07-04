import pandas as pd
import yaml
from numpy import isnan


class WashingFilter:
    def __init__(self, configPath):
        configStream = open(configPath, 'r')
        self.config = yaml.safe_load(configStream)
        configStream.close()
        self.record = pd.DataFrame(columns=['cleanCounter', 'dirtyCounter'])
        self.ladyRestroomTrigger = self.__virtualFence((1,1), self.config['Fences']['lady'])
        self.ladyIOTrigger = self.__edgeTrigger()
        self.menRestroomTrigger = self.__virtualFence((1,1), self.config['Fences']['men'])
        self.menIOTrigger = self.__edgeTrigger()
        self.sinkTrigger = self.__virtualFence((0,1), self.config['Fences']['sink'])
        self.checkWashingTrigger = self.__checkAction((self.config['Threshold']['aspect-ratio']))
        self.cleanThreshold = self.config['Threshold']['clean']
        self.dirtyThreshold = self.config['Threshold']['dirty']
        self.waitingThreshold = self.config['Threshold']['waiting']

    def work(self, inputs):
        # inputs are list of object data with its properties
        # all Trigger Functions define below would filtrate input
        # and return custom result
        ladyTable = self.ladyRestroomTrigger(inputs)
        ladyIOTable = self.ladyIOTrigger(inputs, ladyTable)
        menTable = self.menRestroomTrigger(inputs)
        menIOTable = self.menIOTrigger(inputs, menTable)
        sinkTable = self.sinkTrigger(inputs)
        sinkTable = self.checkWashingTrigger(inputs, sinkTable)

        # Step 0: 迭代這一輪的dirtyCounter
        self.record.iloc[:]['dirtyCounter'] += 1

        # Step 1: 對洗手的對象record更新Counter
        # 考慮到計算雜訊，當再次洗手時，補足洗手時間
        # 能減少斷斷續續的洗手偵測結果
        # 只有持續不洗手（累積dirtyCounter）才視為離開狀態了
        for i, obj in enumerate(inputs):
            # pid 作為record的index，直接索引
            pid = obj.track_id
            if sinkTable[i] and pid in self.record.index:
                if isnan(self.record.loc[pid, 'cleanCounter']):
                    self.record.loc[pid, 'cleanCounter'] = 0
                    self.record.loc[pid, 'dirtyCounter'] = 0
                else:
                    self.record.loc[pid, 'cleanCounter'] += self.record.loc[pid, 'dirtyCounter']
                    self.record.loc[pid, 'dirtyCounter'] = 0

        # Step 2: 維護出廁所對象的record
        # 主要有兩部分:
        # A: 出廁所（狀態變換，由IOTable定義），記錄進record
        # B: 出廁所後仍在廁所檢查區域者，不被視為離開，則record的dirtyCounter會保持0
        for i, obj in enumerate(inputs):
            pid = obj.track_id
            if ladyIOTable[i] or menIOTable[i]:
                self.record.loc[pid] = [None, 0]

        # Step 3: 最終檢查record的步驟
        # 會以Counter來過濾出
        #   (1). 超時不洗手的人 => 視為離開了
        #   (2). 洗手時間不滿足的人 => 視為不確實洗手
        #   (3). 洗手時間滿足的人 => 視為已經洗完手了
        if self.record.size == 0:
            return [], [], []
        notWash = isnan(self.record['cleanCounter']) & (self.record['dirtyCounter'] >= self.dirtyThreshold)
        notWashIds = self.record.loc[notWash].index
        self.record = self.record.loc[~notWash]

        wrongWash = ~isnan(self.record['cleanCounter']) & (self.record['dirtyCounter'] >= self.waitingThreshold)
        wrongWashIds = self.record.loc[wrongWash].index
        self.record = self.record.loc[~wrongWash]

        correctWash = self.record['cleanCounter'] >= self.cleanThreshold
        correctWashIds = self.record.loc[correctWash].index
        self.record = self.record.loc[~correctWash]
        return notWashIds, wrongWashIds, correctWashIds

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