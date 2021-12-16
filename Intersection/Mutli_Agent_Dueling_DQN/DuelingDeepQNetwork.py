import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, input_dims, name, checkpoint_dir):
        super(DuelingDeepQNetwork, self).__init__()

        # 3 convolutional layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.conv_output_dims = self.get_conv_output_dimensions(input_dims)

        # 2 fully-connected layers
        self.fc1 = nn.Linear(self.conv_output_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.Value = nn.Linear(512, 1)
        self.Advantage = nn.Linear(512, n_actions)

        # Initialize optimizer and loss functions
        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        # sets device - 'cuda:0' for gpu or 'cpu' for cpu
        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(checkpoint_dir, name)
        print(self.checkpoint_name)
        print(self.parameters)

    def get_conv_output_dimensions(self, input_dims):
        """
        Returns the product of output dimensions of convoluted output to feed
        in linear classifier.
        """
        temp = torch.zeros(1, *input_dims)
        dim1 = self.conv1(temp)
        dim2 = self.conv2(dim1)
        dim3 = self.conv3(dim2)
        return int(np.prod(dim3.size()))

    def forward(self, data):
        """
        Feed forward the network to get the value, advantage tuple
        """
        shared_conv_layer1 = F.relu(self.conv1(data))
        
        # For the Agent_1
        conv_layer2_1 = F.relu(self.conv2(shared_conv_layer1))
        conv_layer3_1 = F.relu(self.conv3(conv_layer2_1))

        output_conv_layer_1 = conv_layer3_1.view(conv_layer3_1.size()[0], -1)

        fc_layer1_1 = F.relu(self.fc1(output_conv_layer_1))
        fc_layer2_1 = F.relu(self.fc2(fc_layer1_1))

        value_1 = self.Value(fc_layer2_1)
        advantage_1 = self.Advantage(fc_layer2_1)
        
        # For Agent_2
        conv_layer2_2 = F.relu(self.conv2(shared_conv_layer1))
        conv_layer3_2 = F.relu(self.conv3(conv_layer2_2))

        output_conv_layer_2 = conv_layer3_2.view(conv_layer3_2.size()[0], -1)

        fc_layer1_2 = F.relu(self.fc1(output_conv_layer_2))
        fc_layer2_2 = F.relu(self.fc2(fc_layer1_2))

        value_2 = self.Value(fc_layer2_2)
        advantage_2 = self.Advantage(fc_layer2_2)
        
        return (value_1, value_2), (advantage_1, advantage_2)

    def save_checkpoint(self):
        """
        Saves the checkpoint to the desired file.
        """
        print('Saving checkpoint...')
        print(self.checkpoint_name)
        torch.save(self.state_dict(), self.checkpoint_name)

    def load_checkpoint(self):
        """
        Loads the checkpoint from the saved file.
        """
        print('Loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_name))
