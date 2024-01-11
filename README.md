# Image Detection Using Nano 33 BLE Arduino Board
## Overview
- This project, focuses on object detection, specifically differentiating between stop signs and non-stop signs.
- Utilizing the Arduino Nano 33 BLE and the OV7675 Camera Module, the project aims to train a binary classification model and deploy it onto an embedded board.
- The dataset combines ten self-derived images and a comprehensive online dataset from Kaggle, tailored to the project's requirements.

## Model Architecture
- The model architecture was refined through experimentation, with a cap at 300k parameters.
- The final model inputs are grayscale images of size 96x96 pixels.
- Various architectures were tested, including MobileNetV1 and models with different parameters and activation layers, leading to the selection of the most effective model in terms of accuracy and parameter count.

## Data and Training
- Custom images include diverse shots of stop signs and similar signs under different conditions to challenge the model's accuracy.
- Data augmentation techniques were applied to increase the diversity of the training data.
- The dataset was segregated into training, testing, and validation sets, with a focus on achieving a balance between stop and non-stop sign images.
- Detailed account of the training process, model choices, and considerations regarding overfitting and dataset composition.

## Results
- The model achieved a training accuracy of 87.06%, validation accuracy of 90.32%, and a test accuracy of 87.05%.
- The model demonstrated a low false rejection rate (0.15%) and false positive rate (0.04%).
- Detailed insights into the model's performance, including its ability to correctly identify images and limitations in detection under certain conditions.
- Graphical representations of loss and accuracy over training epochs are included.
