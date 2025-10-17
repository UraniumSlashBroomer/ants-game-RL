#pragma once
#include <iostream>
#include <vector>

using namespace std;

struct v2i {
    int x;
    int y;
};

const v2i INITIAL_POS = {0, 0};
const int UNIT_MAX_WEIGHT = 5;
const int UNIT_VIS_RADIUS = 3;
const int UNIT_MOVE_POINTS = 2;

const int WORLD_WIDTH = 30;
const int WORLD_HEIGHT = 30;

const int NUMBER_OF_UNITS = 5;

struct Tile {
    v2i pos;

    int move_cost;
    int food_weight;
};

struct Unit {
    v2i pos;

    int current_weight;
    int max_weight;
    
    int vis_radius;
    int move_points;
};

struct WorldState {
    int width;
    int height;
    vector<vector<Tile>> tiles;
    vector<Unit> units;
};

struct Action {
    int unit_index;
    v2i delta;
};

void applyAction(WorldState& state, const Action& action);

Tile generateOneTile(v2i pos, int move_cost, int food_weight);
vector<vector<Tile>> generateTiles(int width, int height);

Unit generateOneUnit();
vector<Unit> generateUnits(int n);

WorldState gameInit();

void printState(WorldState state);
