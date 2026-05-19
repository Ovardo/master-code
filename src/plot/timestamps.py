import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from data_loader import VictoriaParkLoader, VictoriaParkLoader1


def main():
    loader1 = VictoriaParkLoader1()
    for i, step in enumerate(loader1.iterate(3)):
        print(f"Step {i}:")
        for odom in step.odometry:
            print(odom)
    
    loader2 = VictoriaParkLoader()
    for i, step in enumerate(loader2.iterate(3)):
        print(f"Step {i}:")
        for odom in step.odometry:
            print(odom)
    

if __name__ == "__main__":
    main()


