import random
from collections import namedtuple, deque

import torch
import tqdm as tqdm
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from log_utils import logger, mean_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

NUM_STATES = 8
NUM_ACTIONS = 5
BATCH_SIZE = 32


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(8, 512)
        self.h2 = nn.Linear(512, 512)
        self.h3 = nn.Linear(512, 512)
        self.h4 = nn.Linear(512, 256)
        self.h5 = nn.Linear(256, 5)

        # Define sigmoid activation and softmax output
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.h1(x)
        x = self.relu(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.h3(x)
        x = self.relu(x)
        x = self.h4(x)
        x = self.relu(x)
        x = self.h5(x)

        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, args):
        """Save a transition"""
        # self.memory.append(Transition(*args))
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent():
    def __init__(self):

        self.epsilon = 1
        # self.epsilon_decay = 0.9995
        self.epsilon_decay = 0.99975
        self.epsilon_min = 0.05
        self.gamma = 0.95

        self.step_counter = 0
        self.update_target_step = 500

        self.memory = ReplayMemory(10_000)

        self.policy_net = Network().to(device)
        self.target_net = Network().to(device)
        self.policy_net.load_state_dict(torch.load("./models_10/policy_net_9500"))
        self.target_net.load_state_dict(torch.load("./models_10/target_net_9500"))
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.epsi_high = 0.9
        self.epsi_low = 0.05
        self.steps = 0
        self.count = 0
        self.decay = 20000
        self.eps = self.epsi_high

        self.log = logger()
        self.log.add_log('real_return')
        self.log.add_log('combined_return')
        self.log.add_log('avg_loss')

    def index_to_action(self, index):
        if index == 0:
            return -3
        elif index == 1:
            return -1
        elif index == 2:
            return 0
        elif index == 3:
            return 1
        else:
            return 3

    def select_action(self, state):
        steps_done = 0
        steps_done += 1
        if random.random() > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(state, device=device).view(1, -1)
                action_index = self.policy_net(state).max(1)[1].view(1, 1)
                return action_index
        else:
            action_index = torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)
            return action_index

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return 0

        transitions = self.memory.sample(BATCH_SIZE)
        # batch = Transition(*zip(*transitions))

        K = BATCH_SIZE
        samples = self.memory.sample(K)
        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.tensor(S0, dtype=torch.float, device=device)
        A0 = torch.tensor(A0, dtype=torch.long, device=device).view(K, -1)
        R1 = torch.tensor(R1, dtype=torch.float, device=device).view(K, -1)
        S1 = torch.tensor(S1, dtype=torch.float, device=device)
        D1 = torch.tensor(D1, dtype=torch.float, device=device)

        r = R1.squeeze()

        a = None
        with torch.no_grad():
            a = self.target_net(S1)
        b = a.max(dim=1)
        c = b[0]
        d = c.detach()
        target_q = r + self.gamma * d * (1 - D1)
        a1 = self.policy_net(S0)
        policy_q = a1.gather(1, A0)
        L = F.smooth_l1_loss(policy_q.squeeze(), target_q.squeeze())
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        return L.detach().item()


        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
        #                               dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        #
        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)
        #
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

# import gym
# from tqdm import tqdm
# env = gym.make('MountainCar-v0')
# current_state = env.reset()
# next_state, reward, done, _ = env.step(0)
#
# torch.manual_seed(0)
#
# agent = DQNAgent()
#
# for ep in tqdm(range(3000)):
#     current_state = env.reset()
#     sum_r = 0
#     mean_loss = mean_val()
#
#     agent.steps += 1
#     print(agent.epsilon)
#
#     agent.epsilon *= agent.epsilon_decay
#     if agent.epsilon < agent.epsilon_min:
#         agent.epsilon = agent.epsilon_min
#
#     done = False
#     for t in range(200):
#         agent.eps = agent.epsi_low + (agent.epsi_high - agent.epsi_low) * (np.exp(-1.0 * agent.steps / agent.decay))
#         agent.steps += 1
#
#         if ep % 50 == 0:
#             env.render()
#
#         if ep % 500 == 0:
#             torch.save(agent.policy_net.state_dict(), f"./models/policy_net_{ep}")
#             torch.save(agent.target_net.state_dict(), f"./models/target_net_{ep}")
#
#         state = torch.tensor(current_state, device=device, dtype=torch.float).unsqueeze(0)
#         Q = None
#         with torch.no_grad():
#             Q = agent.policy_net(state)
#         num = np.random.rand()
#         if num < agent.epsilon:
#             action = torch.randint(0, Q.shape[1], (1,)).type(torch.LongTensor)
#         else:
#             action = torch.argmax(Q, dim=1)
#
#         next_state, reward, done, _ = env.step(action.item())
#         # reward += current_state[0] + abs(current_state[1])
#         sum_r = sum_r + reward
#
#         agent.memory.push((current_state, action, reward, next_state, done))
#         loss = agent.optimize_model()
#         mean_loss.append(loss)
#         current_state = next_state
#
#         agent.step_counter = agent.step_counter + 1
#         if agent.step_counter > agent.update_target_step:
#             agent.target_net.load_state_dict(agent.policy_net.state_dict())
#             agent.step_counter = 0
#             # print('updated target model')
#         if done:
#             break
#
#         agent.log.add_item('real_return', sum_r)
#         agent.log.add_item('combined_return', sum_r)
#         agent.log.add_item('avg_loss', mean_loss.get())
#
#     print('epoch: {}. return: {}'.format(ep, np.round(agent.log.get_current('real_return')), 2))
#
#
# # states = torch.rand((10, 2), device=device)
# # next_states = torch.rand((10, 2), device=device)
# # actions = torch.randint(3, (10, 1), device=device)
# # rewards = torch.ones((10, 1), device=device) * -1
# #
# # dones = [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
# # dones = torch.tensor([True if d==1 else False for d in dones], dtype=torch.bool, device=device)
# #
# # for i in range(10):
# #     agent.memory.push(states[i], actions[i], next_states[i], rewards[i], dones[i])
# #
# # agent.optimize_model()