import gym # type: ignore
import numpy as np # type: ignore
import random # type: ignore
import time # type: ignore
env = gym.make('Taxi-v3')

ALPHA = 0.9  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 10000
MAX_STEP = 100  # Maximum number of steps per episode

STATES = env.observation_space.n
ACTIONS = env.action_space.n
q_table = np.zeros((STATES, ACTIONS))


def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])


# Training the agent
for episode in range(EPISODES):
    print("Train EPISODE: ", episode)
    
    state = env.reset()[0]  
    done = False
    
    for step in range(MAX_STEP):
        action = choose_action(state)
        next_state, reward, done,_, info = env.step(action)
        
        # Update Q-value
        q_table[state, action] = (1 - ALPHA) * q_table[state, action] + ALPHA * (
            reward + GAMMA * np.max(q_table[next_state, :]) 
        )
        
        state = next_state
        
        if done:
            break
    
    
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
env.close()


# Rendering the environment
env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state = env.reset()[0] 
    done = False
    print("EPISODE: ", episode)
    
    for step in range(MAX_STEP):
        print(f"Step {step+1}, Current state: {state}")
        
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done,_, info = env.step(action)
        state = next_state
        
        if done:
            env.render()
            print(f"Finished after {step+1} steps with reward {reward}")
            break

env.close()
