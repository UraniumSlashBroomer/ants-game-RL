import environment
import visualization as vis
import model
import pygame
import sys
import utils
import torch
import numpy as np
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    config = utils.load_config('config.yaml')

    counter = 0

    env = environment.Environment(**config['env']['map'])
    state = utils.get_state(env)

    actor_hidden_dim = config['models']['actor']['hidden_state_dim']
    critic_hidden_dim = config['models']['critic']['hidden_state_dim']

    actor = model.PolicyModel(state_dim=len(state),
                        hidden_dim=actor_hidden_dim,
                        actions_dim=len(env.units[0].possible_actions)
                        )

    critic = model.CriticModel(state_dim=len(state),
                               hidden_dim=critic_hidden_dim)

    optimizer_actor_lr = config['models']['actor']['lr']
    optimizer_critic_lr = config['models']['critic']['lr']

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=optimizer_actor_lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=optimizer_critic_lr)
    
    try:
        answr = int(input(f'Начать обучение?\n2 - Yes\n1 - No (default)\n0 - Ручное управление\n'))
    except TypeError or ValueError:
        answr = 0

    try:
        answr2 = int(input(f'Дообучить другую модель?\n1 - Yes\n0 - No (default)\n'))
    except TypeError or ValueError:
        answr2 = 0

    if answr2:
        actor, critic = utils.load_checkpoint(actor, critic, 'checkpoint.pth')

    if answr == 2:
        # logs
        file = open('logs.txt', 'w')
        file.write(f'training params:\nstate_dim: {len(state)}, actor hidden state: {actor_hidden_dim}, critic hidden state: {critic_hidden_dim}\n')
        file.write(f'optimizer actor lr: {optimizer_actor_lr}, optimizer critic lr: {optimizer_critic_lr}\n')
        file.write(f'gamma: {config['ppo']['gamma']}, num iters (episodes): {config['train']['num_episodes']}, num timestamps (horizon): {config['train']['horizon']}, num epochs: {config['train']['num_epochs']}.\n\n')
        file.write(f'env params:\nwidth {config['env']['map']['width']}, height: {config['env']['map']['height']}, spawn coords: {config['env']['map']['spawn_coords']}, num units: {config['env']['map']['num_units']}\n\n')
        file.write(f'date: {datetime.now().date()}, time: {datetime.now().time()}\n\n')
        
        actor, critic = model.train_model(actor=actor,
                        critic=critic,
                        optimizer_actor=optimizer_actor,
                        optimizer_critic=optimizer_critic,
                        env=env,
                        **config['train'],
                        **config['ppo'],
                        device=device,
                        logs_file=file)
        file.close()

    elif answr == 0:
        env.reset()
        vis.draw_env(env)
        with torch.no_grad():
            while True:
                for event in pygame.event.get():
                    action_ind = -1

                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_u:
                            action_ind = 0
                        elif event.key == pygame.K_o:
                            action_ind = 1
                        elif event.key == pygame.K_j:
                            action_ind = 2
                        elif event.key == pygame.K_l:
                            action_ind = 3
                        elif event.key == pygame.K_i:
                            action_ind = 4
                        elif event.key == pygame.K_k:
                            action_ind = 5
                    
                    if action_ind != -1:
                        state = utils.get_state(env)
                        action = env.units[0].possible_actions[action_ind]
                        reward = utils.release_action_and_get_reward(env, action)
                        print(reward, action.__name__)
                        done = utils.is_episode_ended(env)
                        if done:
                            print(f'episode ended')
                            env.reset()

                vis.draw_env(env)
                pygame.display.flip()
                
    try:
        answr = int(input(f'Сохранить модели?\n1 - Yes\n0 - No (default)\n'))
    except TypeError:
        answr = 0

    if answr:
        results = {'actor_dict_state': actor.state_dict(),
                   'critic_dict_state': critic.state_dict(),
                   'env_params': config['env'], 
                   'training_params': config['train'],
                   }

        torch.save(results, 'checkpoint.pth')
    
    try:
        answr = int(input(f'Запустить симуляцию в 1 FPS?\n1 - Yes\n0 - No (default)\n'))
    except TypeError:
        answr = 0

    if answr:
        env.reset()
        actor, critic = utils.load_checkpoint(actor, critic, 'checkpoint.pth')
        vis.draw_env(env)
        FPS = 1
        clock = pygame.time.Clock()
        actor.eval()
        critic.eval()
        with torch.no_grad():
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                clock.tick(FPS)
                vis.draw_env(env)
                pygame.display.flip()
                
                for unit in env.units:
                    state = utils.get_state(env)

                    log_prob, action_ind = utils.get_log_proba_and_action_ind(actor, state, unit.possible_actions, 'eval', device)
                    
                    action = unit.possible_actions[action_ind]
                    reward = utils.release_action_and_get_reward(env, action)
                    print(reward, np.exp(log_prob), action.__name__)
                done = utils.is_episode_ended(env)
                if done:
                    env.reset()
