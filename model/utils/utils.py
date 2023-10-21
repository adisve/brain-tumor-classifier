from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNetV2, ResNet50
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class Utils():
    
    @staticmethod
    def get_subfolders(directory):
        return [f.name for f in os.scandir(directory) if f.is_dir()]

    @staticmethod
    def display_images_from_subfolders(training_dir, subfolders):
        plt.figure(figsize=(12, 8))
        for i, subfolder in enumerate(subfolders):
            image = Utils.get_first_image_from_subfolder(os.path.join(training_dir, subfolder))
            if image is not None:
                plt.subplot(1, len(subfolders), i + 1)
                plt.title(subfolder)
                plt.axis('off')
                plt.imshow(image)
        plt.show()

    @staticmethod
    def load_data(path, labels):
        X, y = [], []
        for label in labels:
            label_path = os.path.join(path, label)
            for file in os.listdir(label_path):
                image = Utils.transform_image_file(os.path.join(label_path, file))
                X.append(image)
                y.append(labels.index(label))
        return np.array(X) / 255.0, y

    @staticmethod
    def build_and_compile_model(architecture, image_size=224, dropout_rate=0.4, learning_rate=0.0001, regularizer_value=0.01, metrics=['accuracy', 'AUC']):
        base_model = None
        match architecture:
            case 'EfficientNet':
                base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
            case 'DenseNet':
                base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
            case 'MobileNet':
                base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
            case 'ResNet50':
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
            case _:
                raise ValueError(f"Unsupported architecture: {architecture}")
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(dropout_rate)(x)
        x = Dense(4, activation='softmax', kernel_regularizer=l2(regularizer_value))(x) # Since we are using pretrained weights we do not need to specify an initializer like GlorotUniform in our layers
        
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                    loss='categorical_crossentropy', 
                    metrics=metrics)
        return model

    @staticmethod
    def display_summaries(models):
        for model in models.values():
            model.summary()
    
    @staticmethod
    def train_models(X_train, y_train, X_val, y_val, models_dict, batch_size=64, epochs=30, min_delta=0.001, patience=5, lr_factor=0.3) -> dict:
        history_dict = {}
        
        early_stopping, scheduler = Utils.init_earlystopping_and_scheduler(min_delta, patience, lr_factor)
        for model_name, model in models_dict.items():
            callbacks = Utils.init_callbacks(model_name, early_stopping, scheduler)
            history_dict[model_name] = Utils.train_model(X_train, y_train, X_val, y_val, model, batch_size, epochs, callbacks)
        return history_dict
    
    @staticmethod
    def train_model(model_name, image_gen, X_train, y_train, X_val, y_val, model, batch_size=64, epochs=30, min_delta=0.001, patience=3, lr_factor=0.3):
        early_stopping, scheduler = Utils.init_earlystopping_and_scheduler(min_delta, patience, lr_factor)
        callbacks = Utils.init_callbacks(model_name, early_stopping, scheduler)
        model.fit(
            image_gen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks
        )
        return model
        
    @staticmethod
    def init_callbacks(model_name, early_stopping, scheduler):
        checkpoint = ModelCheckpoint(
            filepath=f'{model_name}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        callbacks = [
            early_stopping,
            scheduler,
            checkpoint,
        ]
        return callbacks
    
    @staticmethod
    def get_first_image_from_subfolder(subfolder_path):
        image_files = os.listdir(subfolder_path)
        if image_files:
            image_path = os.path.join(subfolder_path, image_files[0])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        return None

    @staticmethod
    def transform_image_file(file, image_size=224):
        image = cv2.imread(file, 0) 
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size))
        return image

    @staticmethod
    def init_earlystopping_and_scheduler(min_delta, patience, lr_factor):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=min_delta,
            patience=patience,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )

        scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=patience,
            verbose=1,
            mode='min'
        )
        
        return early_stopping, scheduler
    
    @staticmethod
    def plot(metric, history):
        plt.figure(figsize=[8,6])
        plt.plot(history.history[metric], 'r', linewidth=3.0)
        plt.plot(history.history[f'val_{metric}'], 'b', linewidth=3.0)
        plt.legend([f'Training {metric}', f'Validation {metric}'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title(f'{metric} Curves', fontsize=16)
        plt.show()