import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from Classes.gui_classes import RootWindow

def main():
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver")
    
    np.set_printoptions(suppress=True)

    print("Launching Regression Model Trainer Setup...")
    RootWindow()

if __name__ == "__main__": main()