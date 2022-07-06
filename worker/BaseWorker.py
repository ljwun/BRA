class BaseWorker:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if not self._conditionWork():
            self._endingWork()
            raise StopIteration
        return self._workFlow()
    
    def _workFlow(self):
        raise NotImplementedError
    def _conditionWork(self):
        '''return true or false'''
        raise NotImplementedError
    def _endingWork(self):
        raise NotImplementedError