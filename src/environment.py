import numpy as np

class Unit:
    def __init__(self, init_coords: np.ndarray, move_points: int, max_weight: int) -> None:
        self._move_points = move_points
        self._max_weight = max_weight
        self._coords = init_coords
        self._current_weight = 0
        self._view_radius = 3

    def pickup(self, weight: int) -> int:
        possible_pickup = min(self._max_weight - self.current_weight, weight)
        self.current_weight += possible_pickup
        
        return weight - possible_pickup

    def move(self, new_coords: np.ndarray) -> np.ndarray:
        self._coords += new_coords 
        
        return self._coords
    
    def check_borders(self, width: int, height: int) -> bool:
        if self._coords[0] < 0 or self._coords >= width or \
                self._coords[1] < 0 or self._coords >= height:
                    return False
        return True


class Tile:
    def __init__(self, init_coords: np.ndarray, move_cost: int, food_weight: int) -> None:
        self._coords = init_coords
        self._move_cost = move_cost
        self._food_weight = food_weight

class Map:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self.tiles = self._generate_tiles()

    def _generate_tiles(self) -> list:
        self.tiles = []
        for i in range(self._width):
            for j in range(self._height):
                coords = np.array((i, j))
                move_cost = np.random.randint(0, 3)
                food_weight = np.random.randint(3, 5) if  np.random.randn() > 0.9 else 0
                
                tile = Tile(coords, move_cost, food_weight)
                self.tiles.append(tile)

        return self.tiles

class Environment:
    def __init__(self, width: int, height: int, spawn_coords: np.ndarray, num_units: int) -> None:
        self.map = Map(width, height)
        self.units = self._generate_units(num_units, spawn_coords)

    def _generate_units(self, num_units: int, spawn_coords: np.ndarray) -> list: 
        self.units = []
        for _ in range(num_units):
            # temporal numbers
            move_points = 5  
            max_weight = 5
            unit = Unit(spawn_coords, move_points, max_weight)
            self.units.append(unit)

        return self.units
