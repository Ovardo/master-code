# Master's Thesis

This repository contains the code developed for **TTK4900 – Master’s Thesis in Engineering Cybernetics** at **NTNU**.
 
Code related to the **Specialization Project (TTK4550)** is available in the branch `project-thesis-baseline`.

## Structure
TODO:
├── simulator/        # Motion and measurement simulation. 
├── factors/          # Factor graph definitions. 
├── optimization/     # Batch optimization routines. 
├── visualization/    # Plotting and (future) video tools. 
├── utils/            # Helper functions and data handling. 
└── experiments/      # Example runs and configurations. 


### TODO:   
- [ ] use struct for sim_data['measurments']  
- [ ] tweak sim_data format in general?  
- [x] handle landmarks in simulator that are never seen  
- [ ] handlesteps when no landmarks are seen  
- [ ] make simulator take control input as input as opposed to explicit poses? Static Simulator --> Dynamic simulator   
- [x] Implement itertive full batch optimization  
- [ ] Incremental factor graph (ISAM)  
- [ ] Visualization of SLAM results, instead of plotting each step, make a video that can be stepped trough.  
- [ ] Implement data association (JCBB)   





