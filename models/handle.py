import time


def handle(t):
    time.sleep(int(t))
    return f"Done. And took {t} seconds"
