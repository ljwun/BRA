from loguru import logger
class BaseWorker:
    def __init__(self, EOI=[]):
        self.events = []
        # event of interest
        self.EOI = EOI

    def __iter__(self):
        return self

    def __next__(self):
        if not self._examWork():
            return None
        dataToWork, decide = self._preparatory()
        if not decide:
            self._conditionWork()
            return None
        return self._workFlow(dataToWork)
    
    def _workFlow(self, data):
        raise NotImplementedError
    def _conditionWork(self):
        raise NotImplementedError
    def _preparatory(self):
        return None, False
    def _endingWork(self):
        raise NotImplementedError
    def _examWork(self):
        raise NotImplementedError
    def _signal(self, event):
        logger.trace(f'{event} is in EOI? {event in self.EOI}')
        if event in self.EOI:
            self.events.append(event)
            logger.trace(f'events after append event : {self.events}')
    def receive_signal(self):
        logger.trace(f'events before retrieve : {self.events}')
        bfr = self.events[:]
        self.events = []
        return bfr