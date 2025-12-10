#!/bin/bash
python test_miou.py --model_path /ws/data/Ref-lerf/train/ramen --name origin_test --iteration 0

python test_miou.py --model_path /ws/data/Ref-lerf/train/ramen --name origin_test --iteration 1

python test_miou.py --model_path /ws/data/Ref-lerf/train/ramen --name origin_test --iteration 2

python test_miou.py --model_path /ws/data/Ref-lerf/train/ramen --name origin_test --iteration 3

python test_miou.py --model_path /ws/data/Ref-lerf/train/ramen --name origin_test --iteration 4

