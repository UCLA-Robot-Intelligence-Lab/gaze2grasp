import time
import zarr
import numpy as np
import os
import argparse
import signal

from ril_env.xarm_env import XArmEnv, XArmConfig
from ril_env.controller import SpaceMouse, SpaceMouseConfig
from xarm.wrapper import XArmAPI

# xarm_cfg = XArmConfig()
# xarm_env = XArmEnv(xarm_cfg)

def go_home():
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_servo_angle(angle=[0, 0, 0, 105, 0, 105, 0], speed=50, wait=True)


arm = XArmAPI("192.168.1.223")
arm.motion_enable(enable=True)  # change to False under Sim!!!!!
arm.set_mode(5)
arm.set_state(state=0)
time.sleep(1)

# arm.vc_set_cartesian_velocity([60, 0, 0, 0, 0, 0])
# time.sleep(1)
# arm.vc_set_cartesian_velocity([0, -60, 0, 0, 0, 0])
# time.sleep(1)
# arm.vc_set_cartesian_velocity([0, 0, 60, 0, 0, 0])
# time.sleep(1)
# arm.vc_set_cartesian_velocity([0, 60, 0, 0, 0, 0])
# time.sleep(1)
# arm.vc_set_cartesian_velocity([0, 0, -60, 0, 0, 0])
# time.sleep(1)
# arm.vc_set_cartesian_velocity([0, -60, 0, 0, 0, 0])
# time.sleep(1)

arm.vc_set_cartesian_velocity([0, 0, 0, 15, 0, 0])
time.sleep(1)
arm.vc_set_cartesian_velocity([0, 0, 0, -15, 0, 0])
time.sleep(1)
arm.vc_set_cartesian_velocity([0, 0, 0, 0, -15, 0])
time.sleep(1)
arm.vc_set_cartesian_velocity([0, 0, 0, 0, 15, 0])
time.sleep(1)
arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 15])
time.sleep(1)
arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, -15])
time.sleep(1)

# arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])

# go_home()