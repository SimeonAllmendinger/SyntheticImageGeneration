from collections import deque
from statistics import variance


class EarlyStopping:
    def __init__(self, opt_early_stopping):
        self.OPT_EARLY_STOPPING = opt_early_stopping
        self.length = self.OPT_EARLY_STOPPING['queue_length']
        self.queue = deque()
        self.stop = False


    def push(self, value):
        if len(self.queue) == self.length:
            self.queue.popleft()
            if  self.variance < self.OPT_EARLY_STOPPING['threshold']:
                self.stop=True
        self.queue.append(value)


    def pop(self):
        if len(self.queue) == 0:
            return None
        return self.queue.pop()


    def peek(self):
        if len(self.queue) == 0:
            return None
        return self.queue[-1]
    
    
    @property
    def variance(self):
        return variance(self.queue)
    
    
if __name__ == '__main__':
    queue = EarlyStopping(10)
    queue.push(1)
    queue.push(2)
    queue.push(3)
    print(queue.pop()) #3
    print(queue.pop()) #2
    print(queue.pop()) #1
    