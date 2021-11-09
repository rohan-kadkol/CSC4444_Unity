#%%

import mlagents
import random
import numpy as np
from mlagents_envs.environment import UnityEnvironment as UE

#%%

env = UE(file_name="4444_project",seed=1,side_channels=[])
env.reset()

#%%

print(list(env.behavior_specs))
behavior_name = list(env.behavior_specs)[0]
print(behavior_name)

#%%

print(type(env.behavior_specs[behavior_name]))
print()
print(env.behavior_specs[behavior_name])
spec = env.behavior_specs[behavior_name]

#%%

print(type(spec.observation_specs))
print()
print(type(tuple(spec.observation_specs[0])))
print(spec.observation_specs[0])

#%%

print(type(spec.action_spec))
print()
print(spec.action_spec)

#%%

decision_steps, terminal_steps = env.get_steps(behavior_name)

#%%

print(decision_steps)
print(list(decision_steps))

#%%

tracked_agent = -1
done = False
episode_rewards = 0

print(type(decision_steps.agent_id))
tracked_agent = decision_steps.agent_id[0]

#%%

action = spec.action_spec.random_action(len(decision_steps))
print(action)
print(type(action.discrete))
print(action.discrete)
# action = action.continuous

#%%

env.set_actions(behavior_name, action)

#%%

from dqn_brain import DQNAgent
from tqdm import tqdm

agent = DQNAgent()

for episode in tqdm(range(1000)):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    current_state = decision_steps.obs[0].reshape(8,)
    tracked_agent = -1 # -1 indicates not yet tracking
    done = False # For the tracked_agent
    episode_rewards = 0 # For the tracked_agent

    agent.epsilon = agent.epsilon * agent.epsilon_decay

    while not done:
        # Track the first agent we see if not tracking
        # Note : len(decision_steps) = [number of agents that requested a decision]
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]

        # Generate an action for all agents
        # action = spec.action_spec.random_action(len(decision_steps))
        chosen_action_int = agent.select_action(current_state)
        action = spec.action_spec.empty_action(len(decision_steps))
        action.add_continuous(np.array([[agent.index_to_action(chosen_action_int)]]))

        # Set the actions
        env.set_actions(behavior_name, action)
        # Move the simulation forward
        env.step()
        # Get the new simulation results
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        next_state = decision_steps.obs[0].reshape(8,)

        # print(decision_steps.obs)
        reward = 0
        if tracked_agent in decision_steps: # The agent requested a decision
            reward = decision_steps[tracked_agent].reward
            done = False
#             print(decision_steps[tracked_agent].reward)
        if tracked_agent in terminal_steps: # The agent terminated its episode
            reward = terminal_steps[tracked_agent].reward
#             print(terminal_steps[tracked_agent].reward)
            done = True
        episode_rewards += reward

        agent.memory.push((current_state, chosen_action_int, reward, next_state, done))
        agent.optimize_model()

    print(f"Total rewards for episode {episode} is {episode_rewards}")

#%%

# env.reset()
#
# #%%
#
# decision_steps, terminal_steps = env.get_steps(behavior_name)
# action = spec.action_spec.random_action(len(decision_steps))
# print(action.discrete)
# print(action.continuous)
#
# #%%
#
# decision_steps, terminal_steps = env.get_steps(behavior_name)
# # action = spec.action_spec.random_action(len(decision_steps))
# action = spec.action_spec.empty_action(len(decision_steps))
# action.add_continuous(np.array([[0]]))
# print(action.discrete)
# print(action.continuous)
#
# #%%
#
# env.close()
#
