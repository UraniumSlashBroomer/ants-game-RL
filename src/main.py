import environment
import visualization as vis
import model
import pygame
import sys
import utils
import torch
import numpy as np
import router
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    config = utils.load_config('config.yaml')

    env = environment.Environment(**config['env']['map'])

    answr = router.ask_int(f'Начать обучение?\n2 - Yes\n1 - No (default)\n0 - Ручное управление\n', default=1)

    if answr == 2:
        actor, critic, optimizer_actor, optimizer_critic = utils.init_models_and_optimizers(config, env)        
        
        # logs
        file = open('logs.txt', 'w')
        utils.write_head_in_log(config, file)

        answr2 = router.ask_int(f'Дообучить другую модель?\n1 - Yes\n0 - No (default)\n', default=0)

        if answr2:
            actor, critic = utils.load_checkpoint(actor, critic, 'checkpoint.pth')

        actor, critic = model.train_model(actor=actor,
                        critic=critic,
                        optimizer_actor=optimizer_actor,
                        optimizer_critic=optimizer_critic,
                        env=env,
                        device=device,
                        logs_file=file,
                        **config['train'],
                        **config['ppo'])
        
        file.close()

        answr = router.ask_int(f'Сохранить модели?\n1 - Yes\n0 - No (default)\n', default=0)

        if answr:
            results = {'actor_dict_state': actor.state_dict(),
                       'critic_dict_state': critic.state_dict(),
                       'env_params': config['env'], 
                       'training_params': config['train'],
                       }

            torch.save(results, 'checkpoint.pth')

    elif answr == 0:
        print(f"\ni - вверх\nj - влево\nk - вниз\nl - вправо\no - поднять\nu - положить")
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
                        elif event.key == pygame.K_r:
                            env.reset()
                    
                    if action_ind != -1:
                        state = utils.get_state(env)
                        action = env.units[0].possible_actions[action_ind]
                        reward = utils.release_action_and_get_reward(env, action)
                        print(f'reward: {reward}, action: {action.__name__}')
                        done = utils.is_episode_ended(env)
                        if done:
                            print(f'episode ended')
                            print('-' * 100)
                            env.reset()

                vis.draw_env(env)
                pygame.display.flip()
    
    answr = router.ask_int(f'Запустить симуляцию в {config['eval_viz']['FPS']} FPS?\n1 - Yes\n0 - No (default)\n', default=0)

    if answr:
        env.reset()
        actor, critic, optimizer_actor, optimizer_critic = utils.init_models_and_optimizers(config, env)        
        actor, critic = utils.load_checkpoint(actor, critic, 'checkpoint.pth')
        vis.draw_env(env)
        FPS = config['eval_viz']['FPS']
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
                    print(f"action: {action.__name__}, reward: {reward}, probability: {np.round(np.exp(log_prob), 4)}")
            
                done = utils.is_episode_ended(env)
                if done:
                    env.reset()
                    print('-' * 100)
