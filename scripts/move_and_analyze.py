from tqdm import tqdm
from pathlib import Path
import shutil

def copy_folder(src, dst):
    src = Path(src)
    dst = Path(dst)
    
    dst = dst / src.name
    dst.mkdir(exist_ok=True, parents=True)

    shutil.copytree(src, dst, dirs_exist_ok=True)


src = Path(r"F:\Luigi\M19\20240419\133356\ball_calibration")
dst = Path(r"E:\temp")

# copy dest to src:
copy_folder(src, dst)



