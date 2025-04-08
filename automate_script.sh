#!/bin/bash

count=0
while [ $count -lt 200 ]
do
    echo "Running preprocess_depth_maps.py... Run $((count + 1)) of 200"
    python preprocess_depth_maps.py

    count=$((count + 1))
    echo "Waiting 5 seconds before restarting..."
    sleep 5
done

echo "All runs completed!"