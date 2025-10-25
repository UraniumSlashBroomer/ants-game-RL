import pygame
import sys

TILE_SIZE = 64
TILE_GAP = 2

UNIT_SIZE = 16

FONT_SIZE = 8


pygame.init()
pygame.font.init()

def draw_units(screen, font, units: list) -> None:
    for unit in units:
        x = unit._coords[0] * (TILE_SIZE + TILE_GAP) + TILE_SIZE / 2
        y = unit._coords[1] * (TILE_SIZE + TILE_GAP) + TILE_SIZE / 2
        
        pygame.draw.circle(screen, (255, 255, 255), (x, y), UNIT_SIZE)
        
def draw_map(screen, font, tiles: list) -> None:
    for tile in tiles:
        x = tile._coords[0] * (TILE_SIZE + TILE_GAP)
        y = tile._coords[1] * (TILE_SIZE + TILE_GAP)

        r = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
        text = font.render(f'f: {tile._food_weight} m: {tile._move_cost}',
                           True,
                           (0, 0, 0))
        
        pygame.draw.rect(screen, (255, 0, 0), r, 0)
        screen.blit(text, (x, y))

def start_visualization(tiles: list, units: list) -> None: 
    font = pygame.font.Font(None, size=12)

    screen = pygame.display.set_mode((1200, 800))
    
    draw_map(screen, font, tiles)
    draw_units(screen, font, units)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
