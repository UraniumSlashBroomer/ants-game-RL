#include <raylib.h>
#include "../headers/render.h"

using namespace std;

void renderTiles(vector<vector<Tile>> tiles) {
    for (int i = 0; i < WORLD_WIDTH; i++) {
        for (int j = 0; j < WORLD_HEIGHT; j++) {
            Tile tile = tiles[i][j];
            v2i screen_pos = {tile.pos.x * (TILE_SIZE + TILE_GAP), tile.pos.y * (TILE_SIZE + TILE_GAP)};

            DrawRectangle(screen_pos.x, screen_pos.y, TILE_SIZE, TILE_SIZE, DARKGRAY);
            DrawText(TextFormat("f%d m%d", tile.food_weight, tile.move_cost), screen_pos.x + 2, screen_pos.y + 2, 8, RED);
            
        }
    }
}

void renderUnits(vector<Unit> units) {
    for (int i = 0; i < units.size(); i++) {
        Unit unit = units[i];
        cout << unit.pos.x << ", " << unit.pos.y << endl;
        Vector2 screen_pos = Vector2Zero();  
        screen_pos.x = unit.pos.x * (TILE_SIZE + TILE_GAP) + TILE_SIZE / 2;
        screen_pos.y = unit.pos.y * (TILE_SIZE + TILE_GAP) + TILE_SIZE / 2;
        DrawPoly(screen_pos, 3, TILE_SIZE / 3.0, 90.0f, LIME);
    }
}

void updateCamera(Camera2D& camera) {
    Vector2 mouseWorldPos = GetScreenToWorld2D(GetMousePosition(), camera);
    // camera.offset = GetMousePosition();
    float scale = 0.05f;
    float velocity = 12.0f * 1 / camera.zoom;

    if (IsKeyDown(KEY_E)) {
        camera.zoom = Clamp(expf(logf(camera.zoom) + scale), 0.25f, 10.0f);
    } else if (IsKeyDown(KEY_Q)) {
        camera.zoom = Clamp(expf(logf(camera.zoom) - scale), 0.25f, 10.0f);
    }

    if (IsKeyDown(KEY_W)) {
        camera.target.y -= velocity;
    }
    if (IsKeyDown(KEY_S)) {
        camera.target.y += velocity;
    }
    if (IsKeyDown(KEY_A)) {
        camera.target.x -= velocity;
    }
    if (IsKeyDown(KEY_D)) {
        camera.target.x += velocity;
    }
}

void renderGame(WorldState world_state) {
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "game render");
    MaximizeWindow();

    Camera2D camera = { 0 };
    camera.zoom = 1.0f;
    camera.offset = Vector2 {WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2};
    camera.target = Vector2 {WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2};

    SetTargetFPS(60);
    
    while (!WindowShouldClose()) {
        
        updateCamera(camera);
        BeginDrawing();
            
            BeginMode2D(camera);

            ClearBackground(RAYWHITE);
            renderTiles(world_state.tiles); 
            renderUnits(world_state.units); 
            EndMode2D();

        EndDrawing();

    }

    CloseWindow();
}
