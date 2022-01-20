'''
See:
 - Figure 8.8 of the book "Reinforcement Learning: An Introduction", by Sutton and Barto
 - http://incompleteideas.net/MountainCar/MountainCar1.cp
'''

import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque



def create_model(env, loss_function, optimizer):
    # Create NN analog to Q table
    init = tf.keras.initializers.HeNormal()
    inputs = tf.keras.Input(shape=env.observation_space.shape)
    l1 = tf.keras.layers.Dense(12, activation='relu', kernel_initializer=init)(inputs)
    l2 = tf.keras.layers.Dense(12, activation='relu', kernel_initializer=init)(l1)
    outputs = tf.keras.layers.Dense(env.action_space.n)(l2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    return model



def training_step(env, model, target_model, replay_memory, LR, DF):
    """"""
    # replay_memory element is [observation, action, reward, next_observation, next_action, done]
    # Check that enough steps are in memory before training
    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return None
    
    minibatch_size = 128
    minibatch = random.sample(replay_memory, minibatch_size)
    current_states = np.array([event[0] for event in minibatch])
    current_q_values = model.predict(current_states)
    new_states = np.array([event[3] for event in minibatch])
    new_q_values = target_model.predict(new_states)
    
    trainX, trainY = [], []
    # Enumerate minibatch
    for index, (observation, action, reward, next_observation, next_action, done) in enumerate(minibatch):
        if not done:
            updated_q_value = reward + DF * new_q_values[index][next_action]
        else:
            updated_q_value = reward

        updated_current_q_values = current_q_values[index]
        updated_current_q_values[action] = (1-LR) * updated_current_q_values[action] + LR * updated_q_value

        trainX.append(observation)
        trainY.append(updated_current_q_values)
    
    model.fit(np.array(trainX), np.array(trainY), batch_size=minibatch_size, verbose=0, shuffle=True)
    
    return None



def epsilon_greedy_policy(env, observation, model, epsilon):
    """Choose action based on epsilon-greedy policy"""
    q_values = model.predict(observation.reshape([1, observation.shape[0]])).flatten()

    # Determine next action with epsilon greedy strategy
    if np.random.random() < 1 - epsilon:
        action = np.argmax(q_values) 
    else:
        action = env.action_space.sample()
    
    return action



# Initialize environment
env = gym.make('CartPole-v1')
env.reset()
print('Observation:', env.observation_space)
print('Action:', env.action_space)

def DQLearning(env, n_episodes=250, max_step=200, max_epsilon=1, min_epsilon=0.01, LR=0.7, DF=0.618):
    
    UPDATE = 100 # UPDATE TARGET MODEL EVERY ... STEPS
    TRAIN = 4 # TRAIN ON A MINIBATCH EVERY ... STEPS
    PRINT = 5 # PRINT MODEL AVERAGE REWARD EVERY ... STEPS
    
    epsilon = 1
    
    # Choose loss function and optimizer
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    
    # Initialize NN and target NN
    model = create_model(env, loss_function, optimizer)   # Used for prediction
    target_model = create_model(env, loss_function, optimizer) # Use for update
    target_model.set_weights(model.get_weights())
    
    # Initialize replay memory object
    replay_memory = deque(maxlen=50_000)
    
    # Progress tracking variables
    avg_episode_reward = 0
    
    global_step_counter = 0
    
    for episode in range(n_episodes):
        
        # Episode initialization
        observation = env.reset()
        done = False
        tot_reward = 0
        
        action = epsilon_greedy_policy(env, observation, model, epsilon)
        
        while (not done):
            
            # Render environment for the last five episodes
            #if episode >= (n_episodes - 10):
            if True:
                env.render()

            next_observation, reward, done, info = env.step(action)
            next_action = epsilon_greedy_policy(env, next_observation, model, epsilon)
            replay_memory.append([observation, action, reward, next_observation, next_action, done])
            tot_reward += reward
            global_step_counter += 1

            if global_step_counter % TRAIN == 0:
                training_step(env, model, target_model, replay_memory, LR, DF)
                
            observation, action = next_observation, next_action
        
        # Train when done
        training_step(env, model, target_model, replay_memory, LR, DF)
        
        avg_episode_reward += tot_reward/PRINT
        
        if global_step_counter>=UPDATE:
            #print('Copying main network weights to the target network weights')
            target_model.set_weights(model.get_weights())
            steps_to_update_target_model = 0
        
        if (episode+1)%PRINT==0:
            print('Episode {} finished. Averaged reward for the {} last episodes {}.'.format(episode+1,
                                                                                             PRINT,
                                                                                             avg_episode_reward))
        if episode+1==n_episodes:
            print('Averaged reward for the last episodes {}.'.format(avg_episode_reward))
        
        if (episode+1)%PRINT==0:
            avg_episode_reward = 0.
        
        epsilon = min_epsilon + (max_epsilon-min_epsilon) * (n_episodes-episode-1)/n_episodes
    
    env.close()
        
    return model



#=========================================================================================================================#
trained_Q_network = DQLearning(env)
