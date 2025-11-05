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
    width = 2
    height = 1
    spawn_coords = (0, 0)
    num_units = 1
    counter = 0

    env = environment.Environment(width=width,
                                  height=height,
                                  spawn_coords=spawn_coords,
                                  num_units=num_units)

    visible_tiles_coords = utils.get_visible_coords(env.units)
    state = utils.get_state(env, visible_tiles_coords)

    actor = model.PolicyModel(state_dim=len(state),
                        hidden_dim=512,
                        actions_dim=len(env.units[0].possible_actions)
                        )

    critic = model.CriticModel(state_dim=len(state),
                               hidden_dim=4096)


    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    
    gamma = 0.9
    num_iters = 25
    num_timestamps = 4096
    epochs = 10
    
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
                        device=device)
    
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
        results = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
        actor.load_state_dict(results['actor_dict_state'])
        critic.load_state_dict(results['critic_dict_state'])
        vis.draw_env(env)
        FPS = 1
        clock = pygame.time.Clock()

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
                state = torch.Tensor([state])

                log_prob, action_ind = utils.get_log_proba_and_action_ind(actor, state, env.units[0].possible_actions, device)
                action = env.units[0].possible_actions[action_ind]
                reward = utils.release_action_and_get_reward(env, action)
                print(reward, np.exp(log_prob), action.__name__)
            done = utils.is_episode_ended(env)
            if done:
                env.reset()
