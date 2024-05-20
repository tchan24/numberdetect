# numberdetect

This project uses CNN to recognize and classify images captured from a webcam. The project includes training the model using images and testing the model using live webcam feed.

## Setup

### Prerequisites

Make sure you have Python installed on your system. You will also need to install the following packages:

- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

### Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory.

```sh
git clone <repository-url>
cd <repository-directory>
```
3. Create a virtual environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
4. Install the required packages
```sh
pip install tensorflow opencv-python-headless numpy matplotlib scikit-learn
```

### Training the model

1. Prepare your dataset. Ensure that your dataset is organized into subdirectories where each subdirectory represents a class and contains images for that class.
2. Update the path variable in the OCR_CNN_Training.py script to point to your dataset directory.
3. Run the training script.
```sh
python OCR_CNN_Training.py
```

### Testing the model

1. Connect a webcam to your computer.
2. Run the testing script.
```sh
python OCR_CNN_Testing.py
```