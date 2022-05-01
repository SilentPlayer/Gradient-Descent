from abc import ABC, abstractmethod

class optimizer(ABC):
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    @abstractmethod
    def apply_gradients(self, grads_and_vars):
        pass
