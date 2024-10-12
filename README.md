# Same Same But DifferNet: Implementation on BTAD Dataset

This repository contains an implementation of the "Same Same But DifferNet" model for semi-supervised defect detection using Normalizing Flows, applied to the BTAD  dataset.

## Dataset

The BTAD dataset is a real-world industrial anomaly detection dataset containing 2,830 images of 3 industrial products showcasing body and surface defects.

## Model Training

The model was trained using the Lightning AI platform with the following parameters:
- Meta epochs: 24
- Sub epochs: 8

## Results

| Class | AUROC (Max) | Epoch of Max AUROC | 
|-------|-------------|---------------------|
| 01    | 0.9815      | 18                  | 
| 02    | 0.8345      | 17                  |
| 03    | 0.9868      | 3                   |

## Original Implementation

This project is based on the paper "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows" by Marco Rudolph, Bastian Wandt, and Bodo Rosenhahn. The original implementation can be found [here](https://github.com/marco-rudolph/differnet).

## License

This project is licensed under the MIT License.

