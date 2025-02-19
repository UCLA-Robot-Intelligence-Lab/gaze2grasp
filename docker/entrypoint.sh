#!/bin/bash

export HOME=/root
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
export QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb
exec "$@"

conda init bash && source ~/.bashrc && conda activate contact_graspnet_env