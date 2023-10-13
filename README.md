# Brain Tumor Classification with ResNet50 and Web Service

This repository contains code for classifying different types of brain tumors using a Convolutional Neural Network (CNN) architecture called ResNet50. It also includes a web service built with FastAPI for real-time inference.

# Table of Contents

- [Brain Tumor Classification with ResNet50 and Web Service](#brain-tumor-classification-with-resnet50-and-web-service)
- [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
- [Setup](#setup)
    - [Install Dependencies](#install-dependencies)
- [Web Service](#web-service)
    - [Running the Service](#running-the-service)
    - [API Endpoints](#api-endpoints)
  - [Model Training and Testing](#model-training-and-testing)
    - [Training](#training)
    - [Evaluation](#evaluation)

## Prerequisites

- Python 3.8 or higher
- pip
- Virtual Environment (recommended)

# Setup

1. Clone the Repository: Clone this repository to a folder of your choice.

    ```bash
    git clone https://github.com/your-github-url.git
    ```

2. Navigate to the Project Folder: Move into the cloned project directory.

    ```bash
    cd your-project-directory
    ```

3. Virtual Environment (Recommended): It's often best to create a virtual environment to isolate package dependencies. To create a virtual environment, run the following command:

    ```bash
    python3 -m venv .venv
    ```

To activate the virtual environment, run:

- Linux/Mac:

    ```bash
    source .venv/bin/activate
    ```

- Windows:

    ```bash
    .venv\Scripts\activate
    ```

### Install Dependencies

Install the necessary packages by running the following command:

```bash
pip install .
```

> This command reads the pyproject.toml file and installs all dependencies.

# Web Service

The web service is built using FastAPI and provides real-time inferences from the trained model.

### Running the Service

To run the web service on your local machine, navigate to the server/ directory and execute:

```bash
uvicorn api:app --reload
```

This will start the FastAPI server and you can access the API documentation at http://127.0.0.1:8000/docs.
### API Endpoints

- Predict: POST /predict/
    - Accepts an MRI image and returns the type of brain tumor.

For detailed documentation, refer to the FastAPI generated documentation at http://127.0.0.1:8000/docs.

## Model Training and Testing

To run the model on your local machine, navigate to the model/ directory and open the Jupyter Notebook file.

### Training

The model uses the following Keras callbacks during training:

- EarlyStopping: To stop training early if no improvement in validation loss.
- ReduceLROnPlateau: To reduce learning rate when a metric has stopped improving.
- ModelCheckpoint: To save the model after every epoch.
- LambdaCallback: Custom callback for additional functionalities (here, displaying the confusion matrix).

### Evaluation

Run the model by executing the Jupyter Notebook. Metrics such as loss, accuracy, and AUC (Area Under the Curve) will be displayed at the end, along with interesting graphs.