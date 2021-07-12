#!/bin/bash

mode=""
user_id=""

if [ "$1" = "" ]; then
echo "Please enter the mode(train, predict, recommend)."

elif [ "$1" = "recommend" -a "$2" = "" ]; then
echo "Please enter user id."

else 
mode=$1
user_id=$2
python run.py --mode "$mode" --user_id "$user_id" mode_run

fi


