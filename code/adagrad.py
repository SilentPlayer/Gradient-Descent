from optimizer import optimizer

class adagrad(optimizer):
    
    def __init__(self, learning_rate):
        self.v = [0] * 2
        super().__init__(learning_rate)
    
    def apply_gradients(self, grads_and_vars):
        i = 0
        for grad, var in grads_and_vars:
            self.v[i] = self.v[i] + grad**2
            var.assign_sub((self.learning_rate*grad)/self.v[i]**(1/2))
            i += 1
