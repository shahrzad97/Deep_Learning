'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
from utils import load_datasets, make_dataloaders
import time  

# function to format time
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

#  main function
def main():
    total_start_time = time.time()

    for class_name in c.class_names:
        print(f"Training model for class: {class_name}")
        class_start_time = time.time()

        train_set, test_set = load_datasets(c.dataset_path, class_name)
        train_loader, test_loader = make_dataloaders(train_set, test_set)
        model = train(train_loader, test_loader, class_name)

        class_end_time = time.time()
        class_training_time = class_end_time - class_start_time
        print(f"Finished training for class: {class_name}")
        print(f"Training time for {class_name}: {format_time(class_training_time)}\n")

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Total training time for all classes: {format_time(total_training_time)}")

if __name__ == "__main__":
    main()