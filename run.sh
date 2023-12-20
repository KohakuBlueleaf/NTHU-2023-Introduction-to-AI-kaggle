#!/bin/bash
seed=3407
for round in {0..3}
do
    for i in {0..15}
    do
        python3.11 ./train_pl.py $seed &
        seed=$((seed+1))
    done
    wait
    echo "round $round done"
done
wait
echo "all done"
