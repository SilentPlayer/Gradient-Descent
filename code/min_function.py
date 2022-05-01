import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import imageio
import os
from batch_gd import batch_gradient_descent
from momentum import momentum
from adagrad import adagrad
from adam import adam

class min_function:

    def __init__(self, function, start, global_min, learning_rate=1, optimizers=["gd", "adam", "adagrad", "momentum"]):
        self.function = function
        self.start = start
        self.global_min = global_min
        self.learning_rate = learning_rate
        self.optimizers = optimizers
        self.colors = [np.random.rand(3,) for _ in range(len(optimizers))]
        self.x_data, self.y_data, self.z_data = [[] for _ in range(len(optimizers))], [[] for _ in range(len(optimizers))], [[] for _ in range(len(optimizers))]
        for i, optimizer in enumerate(optimizers):
            opt = self.get_optimizer(optimizer)
            xx, yy, zz = self.minimize(opt)
            self.x_data[i].extend(xx)
            self.y_data[i].extend(yy)
            self.z_data[i].extend(zz)

    def minimize(self, optimizer):
        x = tf.Variable(self.start[0])
        y = tf.Variable(self.start[1])
        x_list, y_list, z_list = [], [], []
        x_list.append(x.numpy())
        y_list.append(y.numpy())
        z_list.append(self.function(x.numpy(), y.numpy()))
        while(not np.allclose(np.array([x.numpy(), y.numpy()]), self.global_min, rtol=1e-01, atol=1e-02)):
            with tf.GradientTape() as tape:
                grads = (g.numpy() for g in tape.gradient(self.function(x, y), [x, y]))
            optimizer.apply_gradients(zip(grads, [x, y]))
            x_list.append(x.numpy())
            y_list.append(y.numpy())
            z_list.append(self.function(x.numpy(), y.numpy()))
        return (x_list, y_list, z_list)

    
    def get_optimizer(self, optimizer):
        if optimizer == 'gd':
            return batch_gradient_descent(self.learning_rate)
        elif optimizer == 'momentum':
            return momentum(self.learning_rate)
        elif optimizer == 'adagrad':
            return adagrad(self.learning_rate)
        elif optimizer == 'adam':
            return adam(self.learning_rate)
        # use keras optimizer if something isn't implemented
        else:
            opt = tf.keras.optimizers.get(optimizer)
            opt.lr = self.learning_rate
            return opt

    def plot2d(self, step=100, x_min=-4, x_max=4, y_min=-4, y_max=4, create_gif=False):
        X = np.linspace(x_min, x_max, 256)
        Y = np.linspace(y_min, y_max, 256)
        XX, YY = np.meshgrid(X, Y)
        Z = self.function(XX, YY)
        plt.figure(dpi=200)
        ax = plt.subplot(111)
        for i in range(len(self.x_data)):
            index = int(step*(len(self.x_data[i])/100))
            ax.plot(self.x_data[i][:index], self.y_data[i][:index], '-', c=self.colors[i])
        ax.set_xlabel('x')
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylabel('y')
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.axis('off')
        ax.legend(self.optimizers)
        ax.contourf(XX, YY, Z, 25, alpha=0.6, cmap=cm.get_cmap('bwr'))
        if not create_gif:
            plt.show()

    def plot3d(self, elev=60, azim=-75, x_min=-3, x_max=3, y_min=-3, y_max=3):
        X = np.linspace(x_min, x_max, 256)
        Y = np.linspace(y_min, y_max, 256)
        XX, YY = np.meshgrid(X, Y)
        Z = self.function(XX, YY)
        fig = plt.figure(dpi=300)
        ax = fig.gca(projection='3d')
        for i in range(len(self.x_data)):
            ax.plot(self.x_data[i], self.y_data[i], self.z_data[i], '-', c=self.colors[i])
        ax.plot_surface(XX, YY, Z, rstride=3, cstride=3, alpha=0.5, cmap=cm.get_cmap('coolwarm'), linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylabel('y')
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlabel('f(x,y)')
        ax.set_zlim(-1, np.max(Z))
        ax.view_init(elev=elev, azim=azim)
        ax.dist=12  
        ax.axis('off')
        ax.legend(self.optimizers)
        plt.show()
        
    def create_gif(self):
        filenames = []
        for i in range(100):
            filename = f'{i}.png'
            filenames.append(filename)
            self.plot2d(i, create_gif=True)
            plt.savefig(filename)
            plt.close()
        with imageio.get_writer('mygif.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(filenames):
            os.remove(filename)
