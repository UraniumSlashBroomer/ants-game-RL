#pragma once
#include <iostream>
#include "raylib.h"
#include "game.h"

const int WINDOW_WIDTH = 640;
const int WINDOW_HEIGHT = 360;

const int TILE_SIZE = 16;
const int TILE_GAP = 2;

void renderUnits();
void renderTiles();
void renderGame(WorldState worldState);
