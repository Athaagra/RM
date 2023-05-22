#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:12:34 2023

@author: Optimus
"""

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
            reward=[1 if self.envidata['value'][steps] > 0 else -1]
        if action==2:
            reward=[1 if self.envidata['value'][steps] < 0 else -1]
        else:
            reward=[1 if self.envidata['value'][steps] == 0 else -1]
        done=[True if len(FinancialData)-1==steps else False]
        #print(done)
        return new_stateenv,reward[0],done,action

import os 
import time 
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from gym import wrappers 
from torch.autograd import Variable 
from collections import deque
#import catch as Catch

save_models=True
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and  not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorach_models")
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage=[]
        self.max_size=max_size
        self.ptr = 0
        
    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)]=transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
    def sample(self,batch_size):
        ind=np.random.randint(0, len(self.storage),size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state,next_state,action,reward,done=self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states),np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        #self.layer_1=nn.Linear(state_dim,400)
        #self.layer_2=nn.Linear(400,300)
        #self.layer_3=nn.Linear(300, action_dim)
        #self.Softmax=nn.Softmax(dim=1)
        self.layer_1=nn.Linear(state_dim,128)
        ##self.layer_2=nn.Linear(400,300)
        self.layer_3=nn.Linear(128, action_dim)
        self.Softmax=nn.Softmax(dim=1)

        self.max_action=max_action
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        ##x = F.relu(self.layer_2(x))
        #x = self.max_action * torch.tanh(self.layer_3(x))
        x = torch.tanh(self.layer_3(x))
        x = self.Softmax(x)
        #print(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + 1, 128)
        #self.layer_2 = nn.Linear(400,300)
        self.layer_3 = nn.Linear(128,1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + 1, 128)
        #self.layer_5 = nn.Linear(400,300)
        self.layer_6 = nn.Linear(128,1)
    
    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        #print(xu)
        x1 = F.relu(self.layer_1(xu))
        #x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        
        x2 = F.relu(self.layer_4(xu))
        #x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2
    def Ql(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        #x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# Selecting the device(CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
    def __init__(self,state_dim,action_dim,max_action):
        self.actor=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters())#params=self.parameters(),lr=7e-3))
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters())#params=self.parameters(),lr=7e-3))
        self.max_action=max_action
    
    def select_action(self,state):
        state=torch.Tensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self,replay_buffer,iterations,batch_size=100,discount=0.99, tau=0.005, policy_noise=0.2,noise_clip=0.5, policy_freq=1):
        
        for b in range(iterations):
            #sample a batch of transition (s,s',a,r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state= torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(np.array(batch_actions)).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done=torch.Tensor(batch_dones).to(device)
            next_action=self.actor_target(next_state)
            noise = torch.Tensor(action).data.normal_(0,policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action=next_action.detach().numpy()#
            noise=noise.detach().numpy()
            noise=np.reshape(noise,(-1,1))
            next_action = next_action + noise#).clamp(-self.max_action, self.max_action)
            next_action = np.argmax(next_action,axis=1)
            next_action=np.reshape(next_action,(-1,1))
            next_action=torch.Tensor(next_action).to(device)
            #print('This is the next action {} this is the next state {}'.format(next_action,next_state))
            #print('This is the next state {} , the next_action {}'.format(len(next_state),len(next_action)))
            target_Q1,target_Q2=self.critic_target(next_state,next_action)
            target_Q=torch.min(target_Q1,target_Q2)
            target_Q=reward + ((1-done) * discount * target_Q).detach()
            #action=np.argmax(action,axis=1)
            action=np.reshape(action,(-1,1))
            #print('This is state {} action {}'.format(state, action))
            #print('This is the state {} , the action {}'.format(len(state),len(action)))
            current_Q1,current_Q2 = self.critic(state,action)
            critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            if b % policy_freq == 0:
                #print('This is the input to actor {}'.format(self.actor(state)))
                selfact=self.actor(state).detach().numpy()#
                selfact = np.argmax(selfact,axis=1)
                selfact=np.reshape(selfact,(-1,1))
                selfact=torch.Tensor(selfact).to(device)
                actor_loss = - self.critic.Ql(state, selfact).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory,filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory,filename))
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

def evaluate_policy(policy,eval_episodes=10):
    avg_reward=0
    for _ in range(eval_episodes):
        obs=env.reset()
        done=False
        while done!= True:
            action = policy.select_action(np.array(list(np.concatenate(obs).flat)))#obs))
            #action=int(np.round(abs(action)))
            print('This is evaluate action {}'.format(action))
            obs,reward,done, _ =env.step(np.argmax(action))
            print(obs,reward,done)
            done=done[0]
            #print(done)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    return avg_reward
#Parameters Initialization
env_name="Risk Management"
seed = 0
start_timesteps=1e4
eval_freq=5e3
max_timesteps=1e6
expl_noise=0.1
batch_size=100
discount=0.99
tau=0.005
policy_noise=0.2
noise_clip=0.5
policy_freq=2

file_name="%s_%s_%s" % ('TD3', env_name, str(seed))
print("Settings: %s" % (file_name))



env = environment()
#state=env.reset()
#done=False
#cumulativeReward=0
#while done!=True:
#    action=np.random.randint(0,3,1)
#    new_state,reward,done=env.step(action)
#    print(new_state,action,reward,done)
#    cumulativeReward+=reward[0]
#    done=done[0]
#    print('This is the cumulative Reward {}'.format(cumulativeReward))
#print('The risk management agent achieves {} percentage'.format(cumulativeReward/len(FinancialData)))
s = env.reset()

s = list(np.concatenate(s).flat)
step_pause = 0.3 # the pause between each plot
#env.render(step_pause) 
# Test
n_test_steps = 100
continuous_execution = False
print_details = False
#env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
#                max_misses=max_misses, observation_type=observation_type, seed=seed)
#env.seed(seed)
torch.manual_seed(seed)
state_dim =len(s) #env.observation_space.shape[0]
action_dim=3#env.action_space.shape[0]
max_action=float(2)#len(env.action_space.high[0])
policy = TD3(state_dim,action_dim,max_action)
replay_buffer=ReplayBuffer()

evaluations=[evaluate_policy(policy)]

total_timesteps=0
timesteps_since_eval=0
episode_num=0
done=True
t0=time.time()

while total_timesteps < max_timesteps:
    if done:
        if total_timesteps !=0:
            print("Total Timesteps:{} Episode Num {} Reward {}".format(total_timesteps,episode_num,episode_reward))
            policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
        
        if timesteps_since_eval >=eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy))
            policy.save(file_name, directory="./pytorach_models")
            np.save("./results/%s" % (file_name), evaluations)
            
        obs=env.reset()
        done=False
        episode_reward=0
        episode_timesteps=0
        episode_num+=1
    if total_timesteps < start_timesteps:
        action = np.random.randint(3)#env.action_space.sample()
    else:
        action = policy.select_action(np.array(obs))
        #action=np.array(int(np.round(abs(action[0]))))
        print('This is the action: {}'.format(action))
        if expl_noise !=0:
            action = (action + np.random.normal(0, expl_noise, size=1)).clip(0, 2)#env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
            #action=np.array(int(np.round(abs(action[0]))))
    #print(action)
    new_obs,reward,done,_=env.step(np.argmax(action))
    print(new_obs,reward,done)
    if done==False:
        done_bool=0
    else:
        done_bool=1
    #done_bool=0 if done==False else done_bool=1#episode_timesteps + 1 == env.max_steps else float(done)#env._max_episode_steps else float(done)
    episode_reward += reward
    replay_buffer.add((list(np.concatenate(obs).flat),list(np.concatenate(new_obs).flat),np.argmax(action), reward, done_bool))
    obs=new_obs
    episode_timesteps +=1
    total_timesteps+=1
    timesteps_since_eval+=1
evaluations.append(evaluate_policy(policy))
if save_models:policy.save('%s' % (file_name),directory="./pytorch_models")
np.save("./results/%s" % (file_name),evaluations)
