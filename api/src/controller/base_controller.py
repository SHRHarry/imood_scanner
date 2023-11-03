# from copy import deepcopy
from threading import Lock
from queue import Queue

class BaseThreadController:
    
    def __init__(self):
        self.process = None
        self.mutex = Lock()
        self.queue = Queue()
    
    def is_running(self):
        return self.process is not None and self.process.is_alive()
    
    # def get_current_status(self):
    #     if self.queue.empty():
    #         status = self.prev_status
    #     else:
    #         status = None
    #         while not self.queue.empty():
    #             status = self.queue.get()
    #         if status:
    #             self.queue.put(status)
    #             self.prev_status = deepcopy(status)
    #         else:
    #             status = self.prev_status
    #     return status
