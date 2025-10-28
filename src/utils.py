import torch
import torch.nn.functional as F

def get_visible_coords(units: list, width: int, height: int) -> set:
    visible_tiles_coords = set()
    for unit in units:
        coords = unit.coords
        for i in range(-unit.view_radius, unit.view_radius):
            for j in range(-unit.view_radius, unit.view_radius):
                new_coords = (coords[0] + i, coords[1] + j)
                if new_coords[0] < 0 or new_coords[1] < 0 or \
                        new_coords[0] >= width or new_coords[1] >= height:
                    pass
                else:
                    visible_tiles_coords.add(new_coords)
    
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
    for coords in visible_tiles_coords:
        tile = env.map.tiles[coords]
        state.append(tile.coords[0])
        state.append(tile.coords[1])
        state.append(tile.food_weight)
        state.append(tile.move_cost)

    for unit in env.units:
        state.append(unit.max_weight)
        state.append(unit.current_weight)
        state.append(unit.coords[0])
        state.append(unit.coords[1])

    spawn_coords = env.map.spawn_coords
    
    state.append(spawn_coords[0])
    state.append(spawn_coords[1])

    state.append(env.map.width)
    state.append(env.map.height)

    return state

"""
possible actions:
move_left, 
move_right,
move_up,
move_down, 
interact
"""

def do_action_and_get_reward(env, action) -> float:
    FOOD_WEIGHT_TO_STOP = 5
    spawn_coords = env.map.spawn_coords
    unit = action.__self__
    if action.__name__ == 'interact':
        unit_food = unit.current_weight
        unit_coords = unit.coords
        tile_food = env.map.tiles[unit_coords].food_weight

        remainder = action(env.map.tiles, spawn_coords)
        if remainder == env.map.tiles[unit.coords].food_weight or (remainder == -1 and unit_food == 0) or (tile_food == 0): # unsuccessful interact
            return -0.5
        else: # successful interact
            if env.map.tiles[spawn_coords].food_weight >= FOOD_WEIGHT_TO_STOP:
                return 10
            return 1
    else: # move action
        unit_coords = unit.coords
        new_unit_coords = action(env.map.width, env.map.height)
        if unit_coords[0] == new_unit_coords[0] and unit_coords[1] == new_unit_coords[1]: # unsuccessful move into the border
            return -0.1

    return -0.01 # successful move, but episode is not over
   
        

def rewards_to_go(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    gammas = torch.arange(rewards.shape[1]) * gamma
    rtg = torch.pow(rewards, gammas).sum()
    return rtg

def calc_advantages(rtg: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return rtg - values

def get_proba(model, states, device):
    if model.__class__.__name__ == 'DumbModel':
        return model.proba
    else:
        if type(states) == list:
            states = torch.Tensor(states)

        model.to(device)
        states.to(device)

        outputs = model(states)
        probabilities = F.softmax(outputs, dim=1)

    return probabilities

def is_episode_ended(env) -> bool:
    FOOD_WEIGHT_TO_STOP = 5
    spawn_coords = env.map.spawn_coords

    if env.map.tiles[spawn_coords].food_weight >= FOOD_WEIGHT_TO_STOP:
        return True
    else: 
        return False
