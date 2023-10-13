# Python in Real-World Applications

## Overview

This project aims to demonstrate the power and flexibility of Python in building a complete machine learning pipeline from data preprocessing to deployment. We use a Convolutional Neural Network (CNN) model to classify medical images of brain tumors and then deploy this model in a web application.

## Tools Used

- Python: The backbone of our project.
- TensorFlow/Keras: For building and training the machine learning model.
- OpenCV: For image processing.
- FastAPI: For back-end API services.
- HTML/JavaScript: For the front-end web application.

## Purpose of the Machine Learning Model

The machine learning model serves as the core engine for automated classification of brain tumors in MRI scans. Specifically, the model is trained to identify various types of brain tumors, such as Glioma, Meningioma, and Pituitary tumors, based on a dataset of annotated images.

## Why is it Important?

Early and accurate diagnosis of brain tumors can significantly improve patient outcomes. Automating this process can also relieve some of the workload on medical professionals, allowing them to focus more on patient care rather than manual image interpretation.

## Architecture Diagram

An architecture diagram is a visual representation of the components or modules within a system and the relationships between them. In the context of machine learning or software development, an architecture diagram usually illustrates how different parts of the system interact with each other, what functionalities they perform, and how data flows between them.

               +---------------+
               |  Web Browser  |
               +-------+-------+
                       |
                       | (User drags image)
                       |
                       v
              +--------+--------+
              |    Frontend     |
              |  (HTML/JS/CSS)  |
              +--------+--------+
                       |
                       | (Sends image via POST)
                       |
                       v
             +---------+----------+
             |      FastAPI       |
             |     Backend API    |
             +---------+----------+
                       |
                       | (Pre-process image)
                       | (Load pre-trained model)
                       | (Make prediction)
                       |
                       v
           +-----------+------------+
           |     Keras ML Model     |
           +------------------------+

## Image Preprocessing

We employ image preprocessing techniques to improve the model's performance. We use OpenCV to apply filters and resize the images.

```py
def augment_image_file(file):
    image = cv2.imread(file, 0) 
    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (image_size, image_size))
    return image
```

## Programming the Machine Learning Model

Building a machine learning model from scratch may sound intimidating, but libraries like TensorFlow and Keras have made this process quite approachable. Here's a breakdown of the programming steps involved.

## Data Loading and Preprocessing

We start by importing necessary Python libraries and loading our dataset. The data consists of medical images, which we preprocess using various techniques like resizing, filtering, and normalization.

```py
import os

def load_data(path, labels):
    X, y = [], []
    for label in labels:
        label_path = os.path.join(path, label)
        for file in os.listdir(label_path):
            image = augment_image_file(os.path.join(label_path, file))
            X.append(image)
            y.append(labels.index(label))
    return np.array(X) / 255.0, y

X_train, y_train = load_data(training_dir, labels)
X_test, y_test = load_data(testing_dir, labels)
```

## Model Architecture

After preprocessing our dataset, the next step is to define the architecture of the machine learning model. For this project, a Convolutional Neural Network (CNN) built on top of a pre-trained ResNet50 architecture is used.

```py
resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size, image_size, 3)
)
model = resnet.output
model = GlobalAveragePooling2D()(model)
model = Dropout(0.4)(model)
model = Dense(4, activation='softmax')(model)
model = Model(inputs=resnet.input, outputs=model)
```

## Model Compilation

Once the architecture is set, we need to compile our model. This involves specifying the optimizer, loss function, and metrics to monitor.

```py
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy', 'AUC']
)
```

## Training the Model

The most critical phase in building a machine learning model is the training stage. This is where the model learns to make predictions by fitting to the provided training data.

```py
history = model.fit(
    image_gen.flow(
        X_train, y_train, batch_size=20
    ),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks
)
```

## Model Evaluation and Saving

After training, we evaluate the model on a test set to measure its performance. If the model performs well, we save it for later use in deployment.

```py
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
print(f'Test Loss: {test_loss}')
print(f'Test AUC: {test_auc}')

model.save('model.keras')
```

By breaking down each of these steps, you can see that programming a machine learning model is a structured process. Each step serves a specific purpose and they all come together to create a robust classification model.

## Deploying the Model with FastAPI

FastAPI allows us to turn our trained model into a web-accessible API. The API accepts an image, preprocesses it, and returns a prediction from the model.

```py
from fastapi import FastAPI, File, UploadFile

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = augment_image_cv2(image)

    array = np.expand_dims(image, axis=0) / 255.0
    prediction = classifier.model.predict(array)
    label = np.argmax(prediction)

    return {"label": labels[label]}
```

## The Web Application

Finally, we create a small front-end web application using HTML and JavaScript, just to visualize how the model can be used practically. The user can upload an image, which gets sent to our FastAPI back end, and receives a prediction.

<img src="web2.png" style="width: 1000px; height: 500px;">

## How it All Fits Together

The front-end allows for user interaction, the FastAPI back-end serves the model, and the model itself is a product of Python programming and machine learning principles.