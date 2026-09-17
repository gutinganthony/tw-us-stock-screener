#!/bin/zsh

APP_DIR="${0:A:h}"
cd "$APP_DIR" || exit 1
python3 server.py --open
