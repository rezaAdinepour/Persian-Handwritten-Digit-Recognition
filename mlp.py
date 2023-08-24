"""
MLP network implemented using pytorch.
"""

import numpy as np
import torch
from sklearn import metrics


class MLP:
    def __init__(self, learning_rate, batch_size, dimensions):
        self.batch_size = batch_size
        self.make_net(learning_rate, dimensions)

    def make_net(self, learning_rate, dimensions):
        input = dimensions[0]
        hidden = dimensions[1]
        out = dimensions[2]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out),
        )
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()

    def train(self, images, labels):
        for i in range(int(len(images)/self.batch_size)):
            batch = images[i*self.batch_size:(i+1) * self.batch_size, :]
            batch_label = np.array(labels)[i*self.batch_size:(i+1) * self.batch_size, :]

            temp = torch.from_numpy(batch.astype(np.float32))
            prediction = self.net(temp)
            loss = self.loss_func(prediction, torch.Tensor(batch_label))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, images, labels):
        prediction = self.net(torch.from_numpy(images.astype(np.float32)))
        loss = metrics.mean_squared_error(labels, prediction.data.numpy())
        return prediction, loss