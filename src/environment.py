import numpy as np

class Unit:
    def __init__(self, init_coords: tuple, move_points: int, max_weight: int) -> None:
        self.move_points = move_points
        self.max_weight = max_weight
        self.coords = init_coords
        self.current_weight = 0
        self.view_radius = 3

        self.possible_actions = [self.lay_down,
                                 self.pickup,
                                 self.move_left,
                                 self.move_right,
                                 self.move_up,
                                 self.move_down]

        self.possible_actions = {i: action for (i, action) in enumerate(self.possible_actions)}
    
    def lay_down(self, tiles: dict, spawn_coords: tuple) -> int:
        # unit drop food at spawn
        if self.coords == spawn_coords:
            tiles[spawn_coords].food_weight += self.current_weight
            self.current_weight = 0
            return 1
        else:
            return -1

    def pickup(self, tiles: dict, spawn_coords: tuple) -> int:
        if self.coords == spawn_coords:
            return -1

        weight = tiles[self.coords].food_weight

        possible_pickup = min(self.max_weight - self.current_weight, weight)
        tiles[self.coords].food_weight -= possible_pickup
        self.current_weight += possible_pickup
        
        return weight - possible_pickup

    def move(self, new_coords: tuple, width: int, height: int) -> tuple:
        old_coords = self.coords
        self.coords = (self.coords[0] + new_coords[0], self.coords[1] + new_coords[1])
        
        if not(self.check_borders(width, height)):
            self.coords = old_coords
        
        return self.coords

    def move_left(self, width: int, height: int) -> tuple:
        return self.move((-1, 0), width, height)

    def move_right(self, width: int, height: int) -> tuple:
        return self.move((1, 0), width, height)

    def move_up(self, width: int, height: int) -> tuple:
        return self.move((0, -1), width, height)

    def move_down(self, width: int, height: int) -> tuple:
        return self.move((0, 1), width, height)

    def check_borders(self, width: int, height: int) -> bool:
        if self.coords[0] < 0 or self.coords[0] >= width or \
                self.coords[1] < 0 or self.coords[1] >= height:
                    return False
        return True


class Tile:
    def __init__(self, init_coords: tuple, food_weight: int) -> None:
        self.coords = init_coords
        self.food_weight = food_weight
        self.move_cost = 1

class Map:
    def __init__(self, width: int, height: int, spawn_coords: tuple, food_to_stop: int, food_scaler: float) -> None:
        self.width = width
        self.height = height
        self.spawn_coords = spawn_coords
        self.food_to_stop = food_to_stop
        self.food_scaler = food_scaler
        self.tiles = self._generate_tiles()

    def _generate_tiles(self) -> dict:
        self.tiles = {}

        mi = self.food_to_stop / (self.width * self.height) * self.food_scaler

        food_weights = np.random.poisson(lam=mi,
                                        size=(self.width, self.height))
        
        food_weights[self.spawn_coords[0], self.spawn_coords[1]] = 0

        remainder = self.food_to_stop - food_weights.sum()
        
        while remainder > 0:
            i, j = np.random.choice(self.width), np.random.choice(self.height)
            
            if i == self.spawn_coords[0] and j == self.spawn_coords[1]:
                continue

            food_weight = np.random.poisson(lam=mi)

            food_weights[i, j] += food_weight
            remainder -= food_weight

        for i in range(self.width):
            for j in range(self.height):
                coords = (i, j)
                
                tile = Tile(coords, food_weights[i, j])
                if coords[0] == self.spawn_coords[0] and coords[1] == self.spawn_coords[1]:
                    tile.move_cost = 0

                self.tiles[coords] = tile

        return self.tiles

class Environment:
    def __init__(self, width: int, height: int, spawn_coords: tuple, num_units: int, food_to_stop: int, food_scaler: float) -> None:
        spawn_coords = tuple(spawn_coords)
        self.map = Map(width, height, spawn_coords, food_to_stop, food_scaler)
        self.units = self._generate_units(num_units, spawn_coords)
        self.num_units = num_units
        self.food_to_stop = food_to_stop
        self.food_scaler = food_scaler

    def _generate_units(self, num_units: int, spawn_coords: tuple) -> list: 
        self.units = []
        for _ in range(num_units):
            # temporal numbers
            move_points = 5  
            max_weight = 5
            unit = Unit(spawn_coords, move_points, max_weight)
            self.units.append(unit)

        return self.units

    def reset(self):
        self.__init__(self.map.width, self.map.height, self.map.spawn_coords, self.num_units, self.food_to_stop, self.food_scaler)

class Memory:
    def __init__(self):
        self.clear()

    def update(self, state, action_index, logprob, reward, done):
        self.states.append(state)
        self.action_indexes.append(action_index)
        self.log_probs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.action_indexes = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
