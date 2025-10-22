#pragma once
#include <iostream>
#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"
#include "game.h"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 450;

const int TILE_SIZE = 32;
const int TILE_GAP = 2;

void renderUnits();
void renderTiles();
void updateCamera(Camera2D& camera);
void renderGame(WorldState worldState);
