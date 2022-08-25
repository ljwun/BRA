class BaseWorker:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        dataToWork, dataToDecide = self._preparatory()
        if not self._conditionWork(dataToDecide):
            self._endingWork()
            raise StopIteration
        return self._workFlow(dataToWork)
    
    def _workFlow(self, data):
        raise NotImplementedError
    def _conditionWork(self, data):
        '''return true or false'''
        raise NotImplementedError
    def _preparatory(self):
        return None, None
    def _endingWork(self):
        raise NotImplementedError

    def GetResultTable(self, clear=True):
        return None