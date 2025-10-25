import numpy as np
import environment
import visualization as vis

if __name__ == "__main__":
    env = environment.Environment(width=30, height=30, spawn_coords=np.array((0, 0)), num_units=3)
    vis.start_visualization(env.map.tiles, env.units)
