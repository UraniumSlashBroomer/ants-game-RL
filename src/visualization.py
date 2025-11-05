import pygame
import utils
import sys

TILE_SIZE = 64
TILE_GAP = 2

UNIT_SIZE = 16

FONT_SIZE = 8


pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((600, 400))
font = pygame.font.Font(None, size=12)

def draw_units(screen, font, units: list) -> None:
    for unit in units:
        x = unit.coords[0] * (TILE_SIZE + TILE_GAP) + TILE_SIZE / 2
        y = unit.coords[1] * (TILE_SIZE + TILE_GAP) + TILE_SIZE / 2
        
        pygame.draw.circle(screen, (255, 255, 255), (x, y), UNIT_SIZE)
        
def draw_map(screen, font, tiles: dict, visible_tiles_coords: list) -> None:
    visible_tiles_coords = set(visible_tiles_coords)
    for tile in tiles.values():
        x = tile.coords[0] * (TILE_SIZE + TILE_GAP)
        y = tile.coords[1] * (TILE_SIZE + TILE_GAP)

        r = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

        if tile.coords in visible_tiles_coords:
            pygame.draw.rect(screen, (255, 0, 0), r, 0)
            text = font.render(f'f: {tile.food_weight} m: {tile.move_cost}',
                           True,
                           (0, 0, 0))

            screen.blit(text, (x, y))
        else:
            pygame.draw.rect(screen, (100, 100, 100), r, 0)


def draw_env(env) -> None: 
    tiles = env.map.tiles
    units = env.units

    visible_tiles_coords = utils.get_visible_coords(units)
    draw_map(screen, font, tiles, visible_tiles_coords)
    draw_units(screen, font, units)
