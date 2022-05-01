from optimizer import optimizer

class adam(optimizer):
    
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.u = [0] * 2
        self.v = [0] * 2
        self.t = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(learning_rate)
    
    def apply_gradients(self, grads_and_vars):
        i = 0
        for grad, var in grads_and_vars:
            self.u[i] = self.beta1 * self.u[i] + (1-self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * grad**2
            u_hat = self.u[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            var.assign_sub((self.learning_rate * u_hat) / (v_hat**(1/2) + self.epsilon))
            i += 1
        self.t += 1
