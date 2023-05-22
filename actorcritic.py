#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 18:38:20 2023

@author: Optimus
"""

import gym
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
#import catch as Catch

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

file_name = 'num'
#data = pd.ExcelFile(file_name +'.xlsx')
data = pd.read_csv(file_name +'.txt', delimiter = "\t")
FinancialData = data[0:2000]#.parse(0, parse_dates=True) #Courses  
#course=pd.read_excel(file_name+'.xlsx', sheet_name='num', parse_dates=False)#\n",
#FinancialDataEnvironment = FinancialData.columns(['adsh','ddate', 'value'])  

class environment:
    def __init__(self):
        self.envidata=FinancialData
        self.stepenv=0
        self.reward=0
        self.done=False
    def reset(self):
        stateenv=self.envidata['ddate'][self.stepenv],int(np.array(self.envidata['adsh'][self.stepenv][0:10]))
        stateenv=np.array(stateenv)
        stateenv=np.reshape(stateenv,(1,2))
        self.stepenv=0
        return stateenv
    def step(self,action):
        steps=self.stepenv
        #print(steps)
        done=self.done
        steps+=1
        self.stepenv=steps
        #print(steps)
        new_stateenv=int(FinancialData['ddate'][steps]),int(np.array(FinancialData['adsh'][steps][0:10]))
        #state=new_state
        new_stateenv=np.reshape(new_stateenv,(1,2))
        if action==1:
            reward=[1 if self.envidata['value'][steps] > 0 else 0]
        if action==2:
            reward=[1 if self.envidata['value'][steps] < 0 else 0]
        else:
            reward=[1 if self.envidata['value'][steps] == 0 else 0]
        done=[True if len(FinancialData)-1==steps else False]
        #print(done)
        return new_stateenv,reward[0],done[0],action
env = environment()
done=False
state=env.reset()
env_video = []
while not done:
    action = np.random.randint(2)#env.action_space.sample()
    state,reward,done, _=env.step(action)
    
class ActorNetwork(keras.Model):
    def __init__(self,n_actions, fc1_dims=512,fc2_dims=256,name='actor',chkpt_dir=''):
        super().__init__()
        self.fc1_dims = fc1_dims
        #self.fc2_dims = fc2_dims 
        self.n_actions = n_actions 
        self.model_name = name 
        self.checkpoint_dir = chkpt_dir 
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac.h5')
        self.fc1=keras.layers.Dense(self.fc1_dims, activation='tanh')
        #self.fc2=keras.layers.Dense(self.fc2_dims, activation='tanh')
        self.pi = keras.layers.Dense(n_actions, activation='softmax')
        
    def call(self,state):
        value=self.fc1(state)
        #value=self.fc2(value)
        pi = self.pi(value)
        return pi

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=256, name='critic', chkpt_dir=''):
            super().__init__()
            self.fc1_dims =fc1_dims 
            #self.fc2_dims = fc2_dims 
            self.n_actions = n_actions 
            self.model_name = name 
            self.checkpoint_dir = chkpt_dir
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac.h5')
            self.fc1 = keras.layers.Dense(self.fc1_dims,activation='tanh')
            #self.fc2 = keras.layers.Dense(self.fc2_dims,activation='tanh')
            self.v = keras.layers.Dense(1, activation=None)
    def call(self,state):
        value = self.fc1(state)
        #value = self.fc2(value)
        #loss='sparse_categorical_crossentropy',
        v=self.v(value)
        return v
class Agent:
        def __init__(self, alpha=0.000003, gamma=0.99, n_actions=2):
            self.gamma=gamma
            self.n_actions = n_actions
            self.action = None
            self.action_space=[i for i in range(self.n_actions)]
            self.actor = ActorNetwork(n_actions=n_actions)
            self.critic = CriticNetwork(n_actions=n_actions)
            self.actor.compile(optimizer=keras.optimizers.SGD(learning_rate=alpha))
            self.critic.compile(optimizer=keras.optimizers.SGD(learning_rate=alpha))
        def choose_action(self, observation):
            state = tf.convert_to_tensor([observation])
            probs = self.actor(state)
            #print('This is the action {}'.format(probs))
            action_probabilities=np.argmax(probs)#tfp.distributions.Categorical(probs=probs)
            action = action_probabilities#action_probabilities.sample()
            #print('This is the action {}'.format(action))
#            log_prob=action_probabilities.log_prob(action)
            self.action=action
            return action#action.numpy()[0]
        def save_models(self):
            print('...saving models ...')
            self.actor.save_weights(self.actor.checkpoint_file)
            self.critic.save_weights(self.critic.checkpoint_file)
            
        def load_models(self):
            print('...loading models...')
            self.actor.load_weights(self.actor.checkpoint_file)
            self.critic.load_weights(self.critic.checkpoint_file)
        
        def learn(self, state, reward, state_, done):
            state = tf.convert_to_tensor([state],dtype=tf.float32)
            state_ =tf.convert_to_tensor([state_],dtype=tf.float32)
            reward = tf.convert_to_tensor(reward,dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                probs = self.actor(state)
                state_value = self.critic(state)
                state_value_ = self.critic(state_)
                state_value = tf.squeeze(state_value)
                state_value_ = tf.squeeze(state_value_)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(self.action)
                delta = reward + self.gamma+state_value_*(1-int(done)) - state_value
                actor_loss =-log_prob*delta
                critic_loss = delta**2
            actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

def plotLearning(scores, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t]=np.mean(scores[max(0,t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig('cumulativeppo.png')
    plt.show()
agent=Agent(alpha=1e-2500,n_actions=3)
n_games = 60
best_score = 0#env.reward_range[0]
score_history = []
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

for i in range(n_games):
    observation=env.reset()
    done=False
    score = 0
    print('this is the game {}'.format(i))
    observation = list(np.concatenate(observation).flat)
    while done!=True:
        action = agent.choose_action(observation)
        #print(action)
        observation_,reward, done, info = env.step(action)
        #print(i,observation_)
        observation_ = list(np.concatenate(observation_).flat)
        score += reward
        if not load_checkpoint:
            agent.learn(observation, reward, observation_,done)
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-10:])
    
    if avg_score > best_score:
        best_score = avg_score 
        if not load_checkpoint:
            agent.save_models()
        if n_games % 50 >= 0:
            print('episode',i,'score %.1f' % score, 'avg_score %.1f' % avg_score)
    if n_games % 50 == 0:
        print('episode', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
if not load_checkpoint:
    x=[i+1 for i in range(n_games)]
    plotLearning(score_history, window=100)

agent.load_models()
done=False
Reward=[]
avg_re=0
rew_ep=[]
for i in range(0,len(FinancialData)):
    state=env.reset()
    state = list(np.concatenate(state).flat)
    done=False
    steps=0
    Reward.append([i,np.mean(rew_ep)])
    rew_ep=[]
    while done!=True:
        action=agent.choose_action(state)
        state,reward,done,_=env.step(action)
        steps+=1
        print(i,action,reward)
        rew_ep.append(reward)
        avg_re=np.mean(rew_ep)
        #if done==True:
        #    print('winning')
            #steps=0
        state = list(np.concatenate(state).flat)

Reward=np.array(Reward)
plt.figure(figsize=(13, 13))
plt.ylabel('Score')
plt.xlabel('Game')
plt.plot(Reward[:,0], Reward[:,1])
plt.savefig('rewardEpActorCritic7.png')
plt.show()
