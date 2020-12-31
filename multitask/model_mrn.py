import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensor_op

class MRN(object):
    def __init__(self,
                 num_conv=3,
                 num_tasks=4,
                 input_dim=(10,30),
                 output_dim=2,
                 lr=0.1,
                 device='cuda'):
        # super().__init__()

        def select_func(x):
            if x > 0.1:
                return 1.0 / x
            else:
                return x
        self.select_func = select_func

        self.device = device
        self.num_tasks = num_tasks

        self.train_cross_loss = 0
        self.train_multitask_loss = 0
        self.train_total_loss = 0
        
        self.print_interval = 50
        self.cov_update_freq = 100

        # add convolutional layers to shared layers module
        self.shared_layers = nn.Sequential()
        in_channels = 1
        out_channels = 64
        for i in range(num_conv):
            self.shared_layers.add_module('conv_layer{}'.format(i+1),
                                          self.conv_layer(in_channels, out_channels))
            in_channels = out_channels
            out_channels = min(2*out_channels, 512)
        
        # calculate output dim of convolutions
        final_dim = torch.tensor(input_dim)
        for i in range(num_conv):
            final_dim = final_dim // 2
        
        # input for fc layers is vector of size (final_dim * # of channels)
        in_features = torch.prod(final_dim).item() * in_channels
        fc_features = 1024
        # add first two fully connected layer to shared layers module
        fc12 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, fc_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(fc_features, fc_features),
            nn.ReLU(),
            nn.Dropout()
        )
        self.shared_layers.add_module('fc12', fc12)
        # move to device
        self.shared_layers = self.shared_layers.to(device)

        # create task-specific layers
        self.weight_size = (fc_features, output_dim)
        self.task_layers = [nn.Linear(*self.weight_size)] * num_tasks
        for layer in self.task_layers:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.constant_(layer.bias, 0.0)
        self.task_layers = nn.Sequential(*self.task_layers)

        # nn.DataParallel shenanigans (IGNORING FOR NOW)

        # group all params to be optimized, with option of using diff lr for each
        params = [{'params': self.shared_layers.parameters()}]
        params += [{'params': self.task_layers[i].parameters() for i in range(num_tasks)}]
        # tune these parameters
        self.optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

        # optim_param input dict shenanigans

        # # initialize covariance matrices for the task fc layers
        self.task_cov = torch.eye(num_tasks, device=device, requires_grad=True)
        self.class_cov = torch.eye(output_dim, device=device, requires_grad=True)
        self.feature_cov = torch.eye(fc_features, device=device, requires_grad=True)

        self.criterion = nn.CrossEntropyLoss()
        self.iter_num = 1

    def optimize(self, inputs, labels_list):
        # update learning rate (as needed)

        # classification loss
        for i in range(self.num_tasks):
            self.task_layers[i].train()
        self.shared_layers.train()
        batch_size = inputs.size(0)

        self.optimizer.zero_grad()
        output_list = [0] * self.num_tasks
        losses = [0] * self.num_tasks
        for i in range(self.num_tasks):
            feature_out = self.shared_layers(inputs)
            # output_list[i] = self.task_layers[i](feature_out.view(feature_out.size(0), -1))
            output_list[i] = self.task_layers[i](torch.flatten(feature_out, 1, -1))
            losses[i] = self.criterion(output_list[i], labels_list[i].squeeze())
        classifier_loss = sum(losses)
     
        # multitask loss
        weight_size = self.task_layers[0].weight.size()
        weights_list = [self.task_layers[i].weight.view(1, weight_size[0], weight_size[1]) for i in range(self.num_tasks)]
        weights_cat = torch.cat(weights_list, dim=0).contiguous()

        multitask_loss = tensor_op.MultiTaskLoss(weights_cat, self.task_cov, self.class_cov, self.feature_cov)
        total_loss = classifier_loss + multitask_loss
        # cumulative cross-entropy and multitask loss for each print iteration
        self.train_cross_loss += classifier_loss.item()
        self.train_multitask_loss += multitask_loss.item()

        total_loss.backward()
        self.optimizer.step()

        if self.iter_num % self.cov_update_freq == 0:
            # get updated weights
            weights_list = [self.task_layers[i].weight.view(1, weight_size[0], weight_size[1]) for i in range(self.num_tasks)]
            weights_cat = torch.cat(weights_list, dim=0).contiguous()
            
            # update cov parameters
            temp_task_cov = tensor_op.UpdateCov(weights_cat.detach(), self.class_cov.detach(), self.feature_cov.detach())

            # eigenvalue decomposition (finding inverse)
            u, s, v = torch.svd(temp_task_cov)
            # s = s.apply_(self.select_func).to(self.device)
            s = torch.where(s > 0.1, 1.0 / s, s)

            self.task_cov = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(self.task_cov)
            if this_trace > 3000.0:
                self.task_cov = (self.task_cov / this_trace * 3000.0).to(self.device)
            else:
                self.task_cov = (self.task_cov).to(self.device)
        
        if self.iter_num % self.print_interval == 0:
            self.train_total_loss = self.train_cross_loss + self.train_multitask_loss
            print('iter {:05d}'.format(self.iter_num))
            print('avg cross-entropy loss: {:.4f}'.format(self.train_cross_loss / float(self.print_interval)))
            print('avg multitask loss: {:.4f}'.format(self.train_multitask_loss / float(self.print_interval)))
            print('avg training loss: {:.4f}'.format(self.train_total_loss / float(self.print_interval)))
            self.train_cross_loss = 0
            self.train_multi_task_loss = 0
            self.train_total_loss = 0
        self.iter_num += 1
    
    # def test(self, input, i):

    def conv_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        return layer

if __name__ == '__main__':
    net = MRN(device='cpu')