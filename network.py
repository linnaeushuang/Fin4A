import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical



class ProcessNetwork(nn.Module):
    def __init__(self, state_dims, mlp_list = [512,256,128], n_conv = 128, conv_kernel = 4, n_fc = 128):
        super(ProcessNetwork, self).__init__()
        # state_dims: [num of fixed param(amount), num of ticke,statedim of ticke, history long]
        self.state_dims = state_dims
        self.vectorOutDim = n_conv
        self.scalarOutDim = n_fc
        self.conv_kernel = conv_kernel
        self.numFcInput = self.scalarOutDim + sum([self.vectorOutDim * (self.state_dims[-1] - self.conv_kernel + 1) for _ in range(self.state_dims[1])])

        self.ticke_nn = nn.ModuleList([nn.Conv1d(self.state_dims[2], self.vectorOutDim, self.conv_kernel) for _ in range(self.state_dims[1])])
        """
        for i in range(self.state_dims[1]):
            self.ticke_nn.append(nn.Conv1d(self.state_dims[2], self.vectorOutDim, self.conv_kernel))
        """

        self.amount_nn = nn.Linear(self.state_dims[0], self.scalarOutDim)
        self.fullyConnectedSequential = nn.Sequential()
        for i in range(len(mlp_list)):
            if i == 0:
                mlp = nn.Linear(self.numFcInput,mlp_list[i])
            else:
                mlp = nn.Linear(mlp_list[i-1],mlp_list[i])
            nn.init.xavier_uniform_(mlp.weight.data)
            nn.init.constant_(mlp.bias.data,0.0)
            self.fullyConnectedSequential.add_module("fc"+str(i),mlp)
            self.fullyConnectedSequential.add_module("ac"+str(i),nn.ReLU(True))

        # ------------ init layer weight ------------
        for tnn in self.ticke_nn:
            nn.init.xavier_uniform_(tnn.weight.data)
            nn.init.constant_(tnn.bias.data,0.0)
        nn.init.xavier_uniform_(self.amount_nn.weight.data)
        nn.init.constant_(self.amount_nn.bias.data,0.0)

    def forward(self, inputs):
        ticke_conv1d_out=[]
        for i,tn in enumerate(self.ticke_nn):
            ticke_conv1d_out.append(F.relu(tn(inputs[:,self.state_dims[0] + i*self.state_dims[2]:self.state_dims[0] + (i+1)*self.state_dims[2],:]),inplace=True).flatten(start_dim=1))
        amount_out = F.relu(self.amount_nn(inputs[:,0:self.state_dims[0],-1]),inplace=True)
        fullyConnectedInput = torch.cat([amount_out]+ticke_conv1d_out,1)
        out = self.fullyConnectedSequential(fullyConnectedInput)
        return out


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, mlp_list = [256, 128],is_continuous_action = True, action_std_init = 0.6):
        super(ActorNetwork, self).__init__()
        self.is_continuous_action = is_continuous_action
        if self.is_continuous_action:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init ** 2)
        self.actor_nn = nn.Sequential()
        for i in range(len(mlp_list)):
            if i == 0:
                mlp = nn.Linear(input_dim, mlp_list[i])
            else:
                mlp = nn.Linear(mlp_list[i-1],mlp_list[i])
            nn.init.xavier_uniform_(mlp.weight.data)
            nn.init.constant_(mlp.bias.data,0.0)
            self.actor_nn.add_module("fc"+str(i),mlp)
            self.actor_nn.add_module("ac"+str(i),nn.ReLU(True))
            if i == len(mlp_list) - 1:
                out_nn = nn.Linear(mlp_list[i],action_dim)
                nn.init.xavier_uniform_(out_nn.weight.data)
                nn.init.constant_(out_nn.bias.data,0.0)
                self.actor_nn.add_module("action",out_nn)
                if not self.is_continuous_action:
                    self.actor_nn.add_module("sm",nn.Softmax(dim=-1))
    def forward(self, state):
        if self.is_continuous_action:
            action_mean = self.actor_nn(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor_nn(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        return action, dist.log_prob(action)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, mlp_list =[256, 128]):
        super(CriticNetwork, self).__init__()
        self.critic_nn = nn.Sequential()
        for i in range(len(mlp_list)):
            if i == 0:
                mlp = nn.Linear(input_dim, mlp_list[i])
            else:
                mlp = nn.Linear(mlp_list[i-1], mlp_list[i])
            nn.init.xavier_uniform_(mlp.weight.data)
            nn.init.constant_(mlp.bias.data,0.0)
            self.critic_nn.add_module("fc"+str(i),mlp)
            self.critic_nn.add_module("ac"+str(i),nn.ReLU(True))
            if i == len(mlp_list) - 1:
                out_nn = nn.Linear(mlp_list[i], 1)
                nn.init.xavier_uniform_(out_nn.weight.data)
                nn.init.constant_(out_nn.bias.data,0.0)
                self.critic_nn.add_module("critic",out_nn)

    def forward(self, state):
        return self.critic_nn(state)

if __name__ == "__main__":
    # state_dims: [num of fixed param(amount), num of ticke,statedim of ticke, history long]
    ticke_state_dim=[2,3,4,7]
    batch_size = 7
    p=ProcessNetwork(ticke_state_dim)
    ticke_state=torch.rand(batch_size,2 + 3 * 4,7)
    print(ticke_state.shape)
    intermediate_data = p(ticke_state)
    print(intermediate_data.shape)
    actor = ActorNetwork(128, 3)
    action, action_prob = actor(intermediate_data)
    print(action.shape,action_prob.shape)
    critic = CriticNetwork(128)
    state_value = critic(intermediate_data)
    print(state_value.shape)

