# fordypning
Code for TTK4550, Engineering Cybernetics Specialization Project at NTNU.  

TODO:   
- [ ] use struct for sim_data['measurments']  
- [ ] tweak sim_data format in general?  
- [x] handle landmarks in simulator that are never seen  
- [ ] handlesteps when no landmarks are seen  
- [ ] make simulator take control input as input as opposed to explicit poses? Static Simulator --> Dynamic simulator   
- [x] Implement itertive full batch optimization  
- [ ] Incremental factor graph (ISAM)  
- [ ] Visualization of SLAM results, instead of plotting each step, make a video that can be stepped trough.  
- [ ] Implement data association (JCBB)   

Done since 16.10:  
- Fixed measurment jacobien with transpose  
- Added incremental full batch optimization loop  


