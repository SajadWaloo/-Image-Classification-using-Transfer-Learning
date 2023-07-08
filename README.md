# Transfer Learning with VGG16 on CIFAR-10 Dataset

This project demonstrates transfer learning using the VGG16 model on the CIFAR-10 dataset. It uses a pre-trained VGG16 model, freezes its layers, and adds custom classification layers on top to perform image classification on the CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. Each image is labeled with one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The dataset is preprocessed by scaling the pixel values between 0 and 1 (float32) to normalize the data before training the model.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python (version 3.6 or later)
- NumPy (```pip install numpy```)
- TensorFlow (```pip install tensorflow```)
- Matplotlib (```pip install matplotlib```)

## Getting Started

To get started with the project, follow these steps:

1. Clone the project repository from GitHub.
2. Install the required dependencies as mentioned in the "Requirements" section.
3. Open a terminal or command prompt and navigate to the project directory.
4. Run the script `transfer_learning.py` using the command: `python transfer_learning.py`.
5. The script will load the CIFAR-10 dataset, preprocess the data, load the pre-trained VGG16 model, freeze its layers, add custom classification layers on top, compile the model, and train the model.
6. During training, the accuracy and loss curves will be plotted using Matplotlib.
7. After training, the model will be evaluated on the test set, and the test accuracy will be displayed in the terminal.

## Transfer Learning with VGG16

The project utilizes the VGG16 model pre-trained on the ImageNet dataset. The pre-trained model is loaded, and its layers are frozen to prevent them from being updated during training. Custom classification layers are added on top of the pre-trained model to adapt it for the CIFAR-10 dataset.

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.

## Results

The project plots the accuracy and loss curves during training using Matplotlib. The curves show the training and validation accuracy/loss over epochs, allowing you to analyze the model's performance.

After training, the model is evaluated on the test set, and the test accuracy is displayed in the terminal.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and adapt the code according to your needs.

If you have any questions or suggestions, please feel free to contact me.

**Author:** Sajad Waloo
**Email:** sajadwaloo786@gmail.com

---
