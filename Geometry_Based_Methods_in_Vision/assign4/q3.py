import numpy as np
import matplotlib.pyplot as plt
import pycolmap
from PIL import Image, ImageOps
from pathlib import Path
import os, shutil


def main():
    output_path = Path("/home/akshay/Downloads/cup/")
    image_dir =  Path("/home/akshay/Downloads/cup/images/")

    if not os.path.exists(output_path):
        #shutil.rmtree(output_path)
        output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    print(maps)
    maps[0].write(output_path)
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    #pycolmap.patch_match_stereo(mvs_path)
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

if __name__ == '__main__':
    main()