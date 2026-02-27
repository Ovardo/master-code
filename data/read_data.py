from scipy.io import loadmat
import numpy as np
from pathlib import Path


def main():
    data_folder = Path(__file__).parents[0]
    mat_path = data_folder.joinpath('victoriaParkDataset.mat')
    data = loadmat(mat_path)
    U = data['controllerInput']
    X_dr = data['deadReckoning']
    X_gps = data['gpsLatLong']
    Z = data['measurements'][:,0]
    print()

if __name__ == "__main__":
    main()