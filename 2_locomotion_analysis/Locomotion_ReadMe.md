**Locomotion Analysis**

The pipeline for locomotion analysis follows these steps: 

1) Labeled regions of interest (the bottoms of the cups) using `1_label_rois.py`. Do this on a computer you can access the GUI from. This saves .pkl files with the location of the ROI, the pixels per cm in x and y directions, and the true FPS of each video.  
2) Calculate distance traveled each second using `2_distance_traveled.py`. This will be the base for all further distance calculations. 
3) Conduct further analyses: e.g. binning distance traveled by larger amounts, calculating velocity, plotting, and statistical analyses. These were all done in `final_analyses.ipynb`. 
