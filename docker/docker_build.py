#!/usr/bin/env python3
import os

if __name__=="__main__":
    cmd = "docker build -t contact_graspnet_env . "
    code = os.system(cmd)