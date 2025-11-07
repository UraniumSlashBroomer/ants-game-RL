import environment
import visualization as vis
import model
import pygame
import sys
import utils
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    width = 4
    height = 4
    spawn_coords = (0, 0)
    num_units = 1
    counter = 0

    env = environment.Environment(width=width,
                                  height=height,
                                  spawn_coords=spawn_coords,
                                  num_units=num_units)

    visible_tiles_coords = utils.get_visible_coords(env.units)
    state = utils.get_state(env, visible_tiles_coords)

    actor_hidden_state = 512
    critic_hidden_state = 2048

    actor = model.PolicyModel(state_dim=len(state),
                        hidden_dim=actor_hidden_state,
                        actions_dim=len(env.units[0].possible_actions)
                        )

    critic = model.CriticModel(state_dim=len(state),
                               hidden_dim=critic_hidden_state)

    optimizer_actor_lr = 3e-4
    optimizer_critic_lr = 1e-4

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=optimizer_actor_lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=optimizer_critic_lr)
    
    gamma = 0.99
    num_iters = 25
    num_timestamps = 4096
    epochs = 5

    # Logs
    file = open('logs.txt', 'w')
    file.write(f'training params:\nstate_dim: {len(state)}, actor hidden state: {actor_hidden_state}, critic hidden state: {critic_hidden_state}\n')
    file.write(f'optimizer actor lr: {optimizer_actor_lr}, optimizer critic lr: {optimizer_critic_lr}\n')
    file.write(f'gamma: {gamma}, num iters (episodes): {num_iters}, num timestamps: {num_timestamps}, num epochs: {epochs}.\n\n')
    file.write(f'env params:\nwidth {width}, height: {height}, spawn coords: {spawn_coords}, num units: {num_units}\n\n')

   
    try:
        answr = int(input(f'Начать обучение? Yes - 1, No - 0\n'))
    except TypeError:
        answr = 0

    if answr:
        actor, critic = model.train_model(actor=actor,
                        critic=critic,
                        optimizer_actor=optimizer_actor,
                        optimizer_critic=optimizer_critic,
                        env=env,
                        gamma=gamma,
                        num_iters=num_iters,
                        num_timestamps=num_timestamps,
                        epochs=epochs,
                        device=device,
                        logs_file=file)
    file.close()

    try:
        answr = int(input(f'Сохранить модели? Yes - 1, No - 0\n'))
    except TypeError:
        answr = 0

    if answr:
        results = {'actor_dict_state': actor.state_dict(),
                   'critic_dict_state': critic.state_dict(),
                   'env_params': {'width': width,
                                  'height': height,
                                  'num_units': 1,
                                  'spawn_coords': spawn_coords},
                   'training_params': {'gamma': gamma,
                                       'num_iters': num_iters,
                                       'num_timestamps': num_timestamps,
                                       'epochs': epochs}
                   }

        torch.save(results, 'checkpoint.pth')

    try:
        answr = int(input(f'Запустить симуляцию в 1 FPS? Yes - 1, No - 0\n'))
    except TypeError:
        answr = 0

    if answr:
        env.reset()
        results = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
        actor.load_state_dict(results['actor_dict_state'])
        critic.load_state_dict(results['critic_dict_state'])
        vis.draw_env(env)
        FPS = 1
        clock = pygame.time.Clock()
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
                    state = utils.get_state(env, visible_tiles_coords)

                    log_prob, action_ind = utils.get_log_proba_and_action_ind(actor, state, env.units[0].possible_actions, 'eval', device)
                    action = env.units[0].possible_actions[action_ind]
                    reward = utils.release_action_and_get_reward(env, action)
                    print(reward, np.exp(log_prob), action.__name__)
                done = utils.is_episode_ended(env)
                if done:
                    env.reset()


