
from pathlib import Path
import sys
# import os


lib_dir = (Path(__file__).parent / '..' ).resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

lib_dir = (Path(__file__).parent / '..' / 'face_detector').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

lib_dir = (Path(__file__).parent / '..' / 'face_recognize').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

print(sys.path)