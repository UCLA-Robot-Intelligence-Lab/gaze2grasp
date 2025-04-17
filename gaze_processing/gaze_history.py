"""
This file contains the implementation of Gaze History

Essentially, we don't want the model to run at every single arbitrary gaze coordinate.
The implementation of this file may change soon, but essentially, my heuristic solution is to wait until the user has been looking at a specific item for x amount of time
only after the user has been staring at the same point (or around the same area buffer) the model analyzes the intention and executes task

In this scenario, a "timestep" refers to the time it takes for a single frame to be rendered/loaded
"""
from collections import deque
import numpy as np

class GazeHistory():
    def __init__(self):
        self.gaze_history = deque([]) # this variable will keep a history of the user's gaze coordinates
        self.last_gaze_stare = (0, 0) # keeps track of the last known stare coordinate

    def log(self, coordinates: tuple) -> bool:
        """
        keep a log of the gaze coordinates of the last 20 timesteps
        if the user is staring at the same object (approximately) for at least 20 timesteps, then execute (return true)
        otherwise, return false
        """
        if len(self.gaze_history) == 20:
            self.gaze_history.popleft()
        self.gaze_history.append(coordinates)
        
        if len(self.gaze_history) < 20: # ignore if it hasn't even been 50 timesteps yet
            return False
        
        np_gaze_history = np.array(self.gaze_history)
        x_coordinates = np_gaze_history[:, 0]
        y_coordinates = np_gaze_history[:, 1]

        # Calculate standard deviation for x and y coordinates
        std_dev_x = np.std(x_coordinates)
        std_dev_y = np.std(y_coordinates)

        # uses standard deviation to tell whether or not the user is staring at the same place
        # the smaller the number, the less deviation the user can make in gaze
        stare = max(std_dev_x, std_dev_y) < 50
        if stare:
            self.last_gaze_stare = self.history_median()
            self.dump_history()
        return stare
    
    def history_median(self) -> tuple:
        """
        from the gaze history, returns the median coordinates (center of the user's gaze/stare)
        """
        print(self.gaze_history)
        if len(self.gaze_history) < 2:
            return None
        if len(self.gaze_history) > 3:
            np_gaze_history = np.array(self.gaze_history)
            x_coordinates = np_gaze_history[:, 0]
            y_coordinates = np_gaze_history[:, 1]

            return [np.median(x_coordinates), np.median(y_coordinates)] # returns coordinates
        else:
            return np.array(self.gaze_history)[-1]
    
    def report_stare_coordinates(self) -> tuple:
        return self.last_gaze_stare

    def dump_history(self):
        self.gaze_history = deque([])
        print("gaze history dumped!")