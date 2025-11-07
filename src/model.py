# TODO loss and finish train function
import numpy as np
import torch
import utils
from torch import nn
import torch.nn.functional as F
import pygame
import visualization as vis
import sys
import environment
from datetime import datetime

def ppo_loss(actor, states: torch.Tensor, actions_indexes: list, old_log_probs: list, advantages: torch.Tensor, clip_epsilon: float = 0.25) -> torch.Tensor:
    old_log_probs_t = torch.Tensor(old_log_probs)
    old_log_probs_t = old_log_probs_t.to(states.device)

    logits = actor(states) # logits [Batch, Action_logits]
    new_log_probabilities = F.log_softmax(logits, dim=-1)[torch.arange(len(logits)), actions_indexes] # new_probablities [Batch, Probabilities]
    
    ratio = torch.exp(new_log_probabilities - old_log_probs_t)

    loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages).mean()

    return loss

def train_model(actor, critic, optimizer_actor, optimizer_critic, env,  gamma, num_iters, num_timestamps, epochs, device, logs_file, visualization=True):
    memory = environment.Memory()
    actor = actor.to(device)
    critic = critic.to(device)

    loss_critic_instance = nn.MSELoss()

    counter = 0
    win_counter = 0
    N = 20
    
    # logs
    logs_file.write(f'date: {datetime.now().date()}, time: {datetime.now().time()}\n\n')

    if visualization:
        FPS = 1024
        clock = pygame.time.Clock()
        vis.draw_env(env)

    for iteration in range(num_iters):
        iter_reward = 0
        for _ in range(num_timestamps):
            counter += 1

            if visualization:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                clock.tick(FPS) 
                vis.draw_env(env)
                pygame.display.flip()

            visible_tile_coords = utils.get_visible_coords(env.units)
            state = utils.get_state(env, visible_tile_coords)
            
            log_prob, action_ind = utils.get_log_proba_and_action_ind(actor, state, env.units[0].possible_actions, 'train', device)
            action = env.units[0].possible_actions[action_ind]
            reward = utils.release_action_and_get_reward(env, action)
            iter_reward += reward
            done = utils.is_episode_ended(env)
            if done:
                win_counter += 1

            if done or counter == N:
                counter = 0
                done = True
                env.reset()

            memory.update(state, action_ind, log_prob, reward, done) 

        with torch.no_grad():
            rtgs = utils.rewards_to_go(memory.rewards, memory.dones, gamma).to(device)
            states = torch.Tensor(memory.states)
            states = states.to(device)
            values = critic(states).squeeze()
            advantages = utils.calc_advantages(rtgs, values)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        actor.train()
        critic.train()
        
        # logs
        logs_file.write(f'iter: {iteration + 1}. Iter reward: {np.round(iter_reward, 2)}. Iter wins: {win_counter}.\n')

        print(f'iter: {iteration + 1}. Iter reward: {np.round(iter_reward, 2)}. Iter wins: {win_counter}.')
        for epoch in range(epochs):
            loss_actor = ppo_loss(actor, states, memory.action_indexes, memory.log_probs, advantages)
            optimizer_actor.zero_grad()
            loss_actor.backward()
            optimizer_actor.step()
            
            values = critic(states).squeeze()
            loss_critic = loss_critic_instance(rtgs, values)
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()
            
            # logs
            logs_file.write(f'Epoch {epoch + 1}/{epochs}. loss actor {loss_actor}, loss critic {loss_critic}\n')
            print(f'Epoch {epoch + 1}/{epochs}. loss actor {loss_actor}, loss critic {loss_critic}')
        
        logs_file.write('\n')

        win_counter = 0

        actor.eval()
        critic.eval()

        memory.clear()

    return actor, critic
            
class DumbModel:
    def __init__(self, possible_actions: list) -> None:
        self.proba = torch.Tensor(np.full(len(possible_actions), 1 / len(possible_actions)))

    def choose_action(self, possible_actions: list):
        return np.random.choice(possible_actions)
    
class PolicyModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, actions_dim: int) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, actions_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class CriticModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
