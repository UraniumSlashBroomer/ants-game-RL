#include <iostream>
#include <random>
#include "../headers/game.h"
using namespace std;

int getRandomInt(int min, int max) {
    static mt19937 generator(random_device{}());

    uniform_int_distribution<int> distribution(min, max);

    return distribution(generator);
}

float getRandn() {
    static mt19937 generator(random_device{}());

    uniform_real_distribution<float> distribution(0.0, 1.0);

    return distribution(generator);
}

Tile generateOneTile(v2i pos) {
    float eps = getRandn();
    int food_weight = 0;
    
    if (eps > 0.9) {
        food_weight = getRandomInt(1, 5);
    }
    
    int move_cost = getRandomInt(1, 2); 

    Tile tile = {pos, move_cost, food_weight};

    return tile;    
}       

Unit generateOneUnit() {
    Unit unit = {INITIAL_POS, 0, UNIT_MAX_WEIGHT, UNIT_VIS_RADIUS, UNIT_MOVE_POINTS};

    return unit;
}

vector<vector<Tile>> generateTiles(int width, int height) {
    vector<vector<Tile>> tiles = {};

    for (int i = 0; i < width; i++) {
        vector<Tile> tiles_row = {};
        
        for (int j = 0; j < height; j++) {
            v2i pos = {i, j};
            Tile tile = generateOneTile(pos);
            tiles_row.push_back(tile);
        }

        tiles.push_back(tiles_row);
    }

    return tiles;
}

vector<Unit> generateUnits(int n) {
    vector<Unit> units = {};
    for (int i = 0; i < n; i++) {
        Unit unit = generateOneUnit();
        units.push_back(unit);
    }

    return units;
} 

WorldState gameInit() {
    WorldState world_state = {};
    world_state.width = WORLD_WIDTH;
    world_state.height = WORLD_HEIGHT;
    world_state.tiles = generateTiles(WORLD_WIDTH, WORLD_HEIGHT);
    world_state.units = generateUnits(NUMBER_OF_UNITS);

    return world_state;
}

void printState(WorldState state) {
    cout << "World width: " << state.width << endl;
    cout << "World height: " << state.height << endl;
    
    for (int i = 0; i < state.width; i++) {
        for (int j = 0; j < state.height; j++) {
            const Tile tile = state.tiles[i][j]; 
            cout << "Tile at pos " << tile.pos.x << ", " << tile.pos.y << endl;
        }
    }
    
    for (int i = 0; i < NUMBER_OF_UNITS; i++) {
        Unit unit = state.units[i];

        cout << "Unit number " << i + 1 << "." << endl;
        cout << "Move points: " << unit.move_points << ", ";
        cout << "Vis radius: " << unit.vis_radius << ", ";
        cout << "Current weight: " << unit.current_weight << ", ";
        cout << "Max possible weight: " << unit.max_weight << "." << endl;
    }
}
