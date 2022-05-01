from optimizer import optimizer

class batch_gradient_descent(optimizer):
    
    def apply_gradients(self, grads_and_vars):
        for grad, var in grads_and_vars:
            var.assign_sub(self.learning_rate * grad)

            
