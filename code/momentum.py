from optimizer import optimizer

class momentum(optimizer):
    
    def __init__(self, learning_rate, beta=0.9):
        self.v = [0] * 2
        self.beta = beta
        super().__init__(learning_rate)
    
    def apply_gradients(self, grads_and_vars):
        i = 0
        for grad, var in grads_and_vars:
            self.v[i] = self.beta * self.v[i] - self.learning_rate * grad
            var.assign_add(self.v[i])
            i += 1
