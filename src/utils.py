import torch
import numpy as np
import environment
import sys
import yaml
import model
from datetime import datetime

def get_visible_coords(units: list) -> list:
    visible_tiles_coords = list()
    for unit in units:
        coords = unit.coords
        for i in range(-unit.view_radius, unit.view_radius):
            for j in range(-unit.view_radius, unit.view_radius):
                new_coords = (coords[0] + i, coords[1] + j)
                visible_tiles_coords.append(new_coords)
    
    return visible_tiles_coords

def get_state(env) -> list:
    visible_tiles_coords = get_visible_coords(env.units)
    """
    state:
    [
        --- N visible tiles ---
        visible_tile.coords.x: int,
        visible_tile.coords.y: int,
        visible_tile.food_weight: int,
        visible_tile.move_cost: int,
        -----------------------
        unit.max_weight: int,
        unit.current_weight: int,
        unit.coords.x: int,
        unit.coords.y: int,
        spawn_coords.x: int,
        spawn_coords.y: int,
        width: int,
        height: int,
    ]
    """
    state = []

    fake_tile = environment.Tile((-1, -1), 0)
    fake_tile.move_cost = 0

    for coords in visible_tiles_coords:
        try:
            tile = env.map.tiles[coords]
        except KeyError:
            tile = fake_tile

        state.append(tile.coords[0] / env.map.width)
        state.append(tile.coords[1] / env.map.height)
        state.append(tile.food_weight / 8)
        # state.append(tile.move_cost)

    for unit in env.units:
        # state.append(unit.max_weight / unit.max_weight)
        state.append(unit.current_weight / unit.max_weight)
        state.append(unit.coords[0] / env.map.width)
        state.append(unit.coords[1] / env.map.height)

    spawn_coords = env.map.spawn_coords
    
    state.append(spawn_coords[0] / env.map.width)
    state.append(spawn_coords[1] / env.map.height)

    # state.append(env.map.width)
    # state.append(env.map.height)

    return state

"""
possible actions:
move_left, 
move_right,
move_up,
move_down, 
lay_down,
puckup
"""

def release_action_and_get_reward(env, action) -> float:
    spawn_coords = env.map.spawn_coords
    unit = action.__self__
    unit_food = unit.current_weight
    unit_coords = unit.coords

    if action.__name__ == 'pickup':
        tile_food = env.map.tiles[unit_coords].food_weight

        remainder = action(env.map.tiles, spawn_coords)
        if remainder == tile_food or (remainder == 0 and tile_food == 0) or (remainder == -1): # unsuccessful pickup
            return -2
        else: # successful pickup
            return 2

    elif action.__name__ == 'lay_down':
        remainder = action(env.map.tiles, spawn_coords)
        if remainder == 1 and unit_food != 0: # successful lay down
            if env.map.tiles[spawn_coords].food_weight >= env.food_to_stop: # finish
                return 25
            elif unit_food != 0:
                return 2
        else: # unsuccessful lay down
            return -2   
    else: # move action
        unit_coords = unit.coords
        new_unit_coords = action(env.map.width, env.map.height)
        if unit_coords[0] == new_unit_coords[0] and unit_coords[1] == new_unit_coords[1]: # unsuccessful move into the border
            return -0.25

    return -0.025 # successful move, but episode is not over
   
def rewards_to_go(rewards: list, dones: list, gamma: float) -> torch.Tensor:
    rtgs = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        G = reward + gamma * G
        rtgs.append(G)
        if done:
            G = 0

    return torch.Tensor(rtgs[::-1])

def calc_advantages(rtg: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return rtg - values

def get_log_proba_and_action_ind(actor, state, possible_actions, mode, device):
    if actor.__class__.__name__ == 'DumbModel':
        return torch.log(actor.proba).item(), np.random.choice(possible_actions.keys())
    else:
        if type(state) == list:
            state = torch.Tensor(state).reshape(1, len(state))

        actor = actor.to(device)
        state = state.to(device)

        with torch.no_grad():
            logits = actor(state)
            dist = torch.distributions.Categorical(logits=logits)
        
        if mode == 'train':
            ind = dist.sample()
        elif mode == 'eval':
            ind = torch.argmax(dist.probs)
        
        log_prob = dist.log_prob(ind)

        try:
            return log_prob.item(), ind.item()
        except NameError:
            print('Error. Mode (train or eval) provided incorrectly')
            sys.exit(1)

def is_episode_ended(env) -> bool:
    spawn_coords = env.map.spawn_coords

    if env.map.tiles[spawn_coords].food_weight >= env.food_to_stop:
        return True
    else: 
        return False

def load_checkpoint(actor, critic, checkpoint: str):
        results = torch.load(checkpoint, map_location=torch.device('cpu'))

        actor.load_state_dict(results['actor_dict_state'])
        critic.load_state_dict(results['critic_dict_state'])

        return actor, critic

def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config

def init_models_and_optimizers(config: dict, env):
    state = get_state(env)

    actor_hidden_dim = config['models']['actor']['hidden_dim']
    critic_hidden_dim = config['models']['critic']['hidden_dim']

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
 
    return actor, critic, optimizer_actor, optimizer_critic

def write_head_in_log(config: dict, file) -> None:
        file.write(f'training params:\nactor hidden state: {config['models']['actor']['hidden_dim']}, critic hidden state: {config['models']['critic']['hidden_dim']}\n')
        file.write(f'optimizer actor lr: {config['models']['actor']['lr']}, optimizer critic lr: {config['models']['critic']['lr']}\n')
        file.write(f'gamma: {config['ppo']['gamma']}, num iters (episodes): {config['train']['num_episodes']}, num timestamps (horizon): {config['train']['horizon']}, num epochs: {config['train']['num_epochs']}.\n\n')
        file.write(f'env params:\nwidth {config['env']['map']['width']}, height: {config['env']['map']['height']}, spawn coords: {config['env']['map']['spawn_coords']}, num units: {config['env']['map']['num_units']}\n\n')
        file.write(f'date: {datetime.now().date()}, time: {datetime.now().time()}\n\n')
 
