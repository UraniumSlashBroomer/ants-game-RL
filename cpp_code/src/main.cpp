#include <iostream>
#include "../headers/game.h"
#include "../headers/render.h"

using namespace std;

int main() {

    WorldState world_state = gameInit();

    renderGame(world_state);

    return 0;
} 
