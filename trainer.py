import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer(object):

    def __init__(self, model, optimizer, config, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.device = device

    def train(self, dataset, criterion, print_every=2):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        start_time = time.time()
        for epoch in range(1, self.num_epoch+1):
            running_loss = 0.0
            loss_count = 0

            for x, t, x_l, _ in loader:
                sorted_l, index = torch.sort(x_l.view(-1),
                                             dim=0, descending=True)
                sorted_x = x[index].to(self.device)
                sorted_t = t[index].to(self.device)
                sorted_t = sorted_t[:, :sorted_l.long().numpy()[0]]
                sorted_l = sorted_l.to(self.device)

                self.optimizer.zero_grad()

                t_hat = self.model(sorted_x, sorted_l)
                loss = criterion(t_hat, sorted_t)

                running_loss += loss.item()
                loss_count += 1

            if epoch % print_every == 0:
                lap_time = time.time() - start_time
                print('epoch: %d, loss: %.3f, %d [sec]'
                      % (epoch, running_loss / loss_count, lap_time))
                start_time = time.time()
