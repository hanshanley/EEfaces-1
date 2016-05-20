#!/bin/bash
while true
do
    echo "Starting the python main"
    echo "Press [CTRL+C] twice to exit properly.."
    # runs the python script, waits for 10 seconds after completing, and repeats
    # TODO: maybe make it sleep for longer? python script sleeps until midnight
    python facerec_main.py -t ~/Dropbox/EEdisplayfaces && sleep 1
    sleep 10
done