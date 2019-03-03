import time
import csv
import sys
import os

i = 0

while True:
    time.sleep(1)
    try:
        with open("../train_logs/training.csv", "w+", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow((i, i+1, i+2))
            print(fp.tell())
            i += 1
    except (KeyboardInterrupt, Exception):
        sys.exit()
