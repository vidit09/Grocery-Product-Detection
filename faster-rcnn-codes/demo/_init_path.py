import os
import sys
import config

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Adding faster RCNN lib 
lib_path = add_path(os.path.join(config.faster_rcnn['path'],'lib'))