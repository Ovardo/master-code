import numpy as np

# -------------------------------
# SIMULATION
# -------------------------------

# Simulation parameters 
simulation_params = {
  'path_type': 'circle', 
  'num_poses': 10,
  'num_landmarks': 15,
  'area_size': 50.0,        # size of the square area (50x50)
  'lidar_max_range': 15.0   # maximum sensor range for landmark observations
}

# Simulation noise paramters
simulation_noise_params = {
    'odom_seed': 42,
    'meas_seed': 42,
    'sigma_x': 0.1,
    'sigma_y': 0.1,
    'sigma_theta': 0.04,
    'sigma_range': 0.1,
    'sigma_bearing': np.deg2rad(1.0),
    'sigma_x0': 0.01,
    'sigma_y0': 0.01,
    'sigma_theta0': 0.001
}

# -------------------------------
# INFERENCE
# -------------------------------

# Inference paramters
inference_params = {
    'init_state': np.array([0.0, 0.0, 0.0]),
    'dead_reckoning': False,
    'association_type': "jcbb",  # "ground_truth", "jcbb" (TODO: todo "maximum_likelihood" "neareast_neighbour", "Constrained Nearnest Neighboor data association https://github.com/ASajwan/isam-vp/tree/master/VictoriaPark") 
    'alpha_individual': 0.999,  # confidence levels for individual compatibility test
    'alpha_joint': 0.9999, # confidence levels for joint compatibility test
    'r_local': simulation_params['lidar_max_range'] + 10, # local feature filtering radius for data association 
    'use_isam': False, # whether to use iSAM2 incremental solver or full batch optimization
    'sensor_offset': np.array([0.0, 0.0]) # np.array([dx, dy]) offset of sensor wrt robot body frame NOTE: not used
}

# Inference noise parameters
inference_noise_params = {
    'sigma_x': 0.1,
    'sigma_y': 0.1,
    'sigma_theta': 0.05,
    'sigma_range': 0.1,
    'sigma_bearing': 0.05,
    'sigma_x0': 0.05,
    'sigma_y0': 0.05,
    'sigma_theta0': 0.05
}