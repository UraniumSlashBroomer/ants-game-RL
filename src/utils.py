import torch
import numpy as np
import environment
import sys

def get_visible_coords(units: list) -> list:
    visible_tiles_coords = list()
    for unit in units:
        coords = unit.coords
        for i in range(-unit.view_radius, unit.view_radius):
            for j in range(-unit.view_radius, unit.view_radius):
                new_coords = (coords[0] + i, coords[1] + j)
                visible_tiles_coords.append(new_coords)
    
    return visible_tiles_coords

def get_state(env, visible_tiles_coords) -> list:
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

    fake_tile = environment.Tile((-1, -1))
    fake_tile.food_weight = 0
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
    FOOD_WEIGHT_TO_STOP = 10
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
            if env.map.tiles[spawn_coords].food_weight >= FOOD_WEIGHT_TO_STOP: # finish
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
    FOOD_WEIGHT_TO_STOP = 10
    spawn_coords = env.map.spawn_coords

    if env.map.tiles[spawn_coords].food_weight >= FOOD_WEIGHT_TO_STOP:
        return True
    else: 
        return False
