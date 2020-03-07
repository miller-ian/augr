import pyrosbag as prb
import time

INTERVAL = 3  # seconds

with prb.BagPlayer("example.bag") as example:
    example.play()