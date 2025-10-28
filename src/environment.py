import numpy as np

class Unit:
    def __init__(self, init_coords: tuple, move_points: int, max_weight: int) -> None:
        self.move_points = move_points
        self.max_weight = max_weight
        self.coords = init_coords
        self.current_weight = 0
        self.view_radius = 3

        self.possible_actions = [self.interact,
                                 self.move_left,
                                 self.move_right,
                                 self.move_up,
                                 self.move_down]

    def interact(self, tiles: dict, spawn_coords: tuple) -> int:
        # unit drop food at spawn
        if self.coords == spawn_coords:
            tiles[spawn_coords].food_weight += self.current_weight
            self.current_weight = 0
            return -1

        # else unit pickup food
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
    def __init__(self, init_coords: tuple, move_cost: int, food_weight: int) -> None:
        self.coords = init_coords
        self.move_cost = move_cost
        self.food_weight = food_weight

class Map:
    def __init__(self, width: int, height: int, spawn_coords: tuple) -> None:
        self.width = width
        self.height = height
        self.spawn_coords = spawn_coords
        self.tiles = self._generate_tiles()

    def _generate_tiles(self) -> dict:
        self.tiles = {}
        for i in range(self.width):
            for j in range(self.height):
                coords = (i, j)
                move_cost = 1
                food_weight = np.random.randint(1, 3) if np.random.randn() > 0.5 else 0
                if coords[0] == self.spawn_coords[0] and coords[1] == self.spawn_coords[1]:
                    move_cost = 0
                    food_weight = 0

                tile = Tile(coords, move_cost, food_weight)
                self.tiles[coords] = tile

        return self.tiles

class Environment:
    def __init__(self, width: int, height: int, spawn_coords: tuple, num_units: int) -> None:
        self.map = Map(width, height, spawn_coords)
        self.units = self._generate_units(num_units, spawn_coords)

    def _generate_units(self, num_units: int, spawn_coords: tuple) -> list: 
        self.units = []
        for _ in range(num_units):
            # temporal numbers
            move_points = 5  
            max_weight = 5
            unit = Unit(spawn_coords, move_points, max_weight)
            self.units.append(unit)

        return self.units
