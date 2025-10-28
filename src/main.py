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
    width = 3
    height = 3
    spawn_coords = (0, 0)
    num_units = 1
    counter = 1
    N = 50 + counter # max length of episode
    reward = 0

    env = environment.Environment(width=width,
                                  height=height,
                                  spawn_coords=spawn_coords,
                                  num_units=num_units)

    visible_tiles_coords = utils.get_visible_coords(env.units, width, height)
    state = utils.get_state(env, visible_tiles_coords)
   
    buffer = []

    model = model.DumbModel(env.units[0].possible_actions)

    vis.draw_env(env)
    FPS = 30
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        clock.tick(FPS)
        vis.draw_env(env)
        pygame.display.flip()
        
        if counter % N == 0:
            env = environment.Environment(width=width,
                                          height=height,
                                          spawn_coords=spawn_coords,
                                          num_units=num_units)
        
        if reward != 10:
            for unit in env.units:
                state = utils.get_state(env, visible_tiles_coords)
                state = torch.Tensor([state])
                probabilities = utils.get_proba(model, state, device)
                
                action = np.random.choice(unit.possible_actions, p=np.array(probabilities).astype(float))
                
                
                reward = utils.do_action_and_get_reward(env, action)
                print(reward)
