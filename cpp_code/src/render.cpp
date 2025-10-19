#include <string>
#include "../headers/render.h"

using namespace std;

void renderTiles(vector<vector<Tile>> tiles) {
    for (int i = 0; i < WORLD_WIDTH; i++) {
        for (int j = 0; j < WORLD_HEIGHT; j++) {
            Tile tile = tiles[i][j];
            v2i screen_pos = {tile.pos.x * (TILE_SIZE + TILE_GAP), tile.pos.y * (TILE_SIZE + TILE_GAP)};

            DrawRectangle(screen_pos.x, screen_pos.y, TILE_SIZE, TILE_SIZE, DARKGRAY);
            DrawText(TextFormat("%d %d", tile.food_weight, tile.move_cost), screen_pos.x, screen_pos.y + TILE_SIZE / 3, 2, RED);
        }
    }
}

void renderUnits(vector<Unit> units) {
    for (int i = 0; i < units.size(); i++) {
        Unit unit = units[i];
        v2i pos = unit.pos;
        v2i screen_pos = {pos.x * (TILE_SIZE + TILE_GAP), pos.y * (TILE_SIZE + TILE_GAP)};
        // DrawPoly;
    }
}
void renderGame(WorldState world_state) {
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "game render");
    MaximizeWindow();

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
    
    BeginDrawing();

    ClearBackground(RAYWHITE);
    renderTiles(world_state.tiles); 
    EndDrawing();

    }

    CloseWindow();
}
