import os
from datetime import datetime


def create_folder(path, safe=True):
    if not os.path.exists(path):
        print("Creating Dirs: {}".format(path))
        os.makedirs(path)
    else:
        print("Dirs: {} Exists".format(path))
        if safe:
            assert False, "Dir exist: {}".format(path)


def get_timestamp():
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S-%f")
    return timestamp
