from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNetV2, ResNet50
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from keras.models import Sequential, Model
from keras.optimizers import Adam
from utils._callbacks import init_callbacks, init_earlystopping_and_scheduler
from typing import List, Tuple, Dict
import numpy as np

def _train_model(model_name: str, image_gen, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, model: Model, batch_size: int = 64, epochs: int = 30, min_delta: float = 0.001, patience: int = 3, lr_factor: float = 0.3) -> Model:
    """Train the model using provided parameters and data."""
    early_stopping, scheduler = init_earlystopping_and_scheduler(min_delta, patience, lr_factor)
    callbacks = init_callbacks(model_name, early_stopping, scheduler)
    model.fit(
        image_gen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )
    return model

def _add_conv_block(model: Sequential, filters: int, kernel_size: Tuple[int, int] = (3, 3), initializer: str = 'he_normal', alpha: float = 0.01) -> None:
    """Add a convolutional block to the model."""
    model.add(Conv2D(filters, kernel_size, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D((2, 2)))

def _add_dense_block(model: Sequential, units: int, initializer: str = 'he_normal', alpha: float = 0.01, dropout_rate: float = 0.5) -> None:
    """Add a dense block to the model."""
    model.add(Dense(units, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

def build_model(num_conv_blocks: int, num_dense_layers: int, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Sequential:
    """Build and compile a new model."""
    model = Sequential()
    
    # Initial convolutional layer
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D((2, 2)))

    filters = 64
    for _ in range(num_conv_blocks - 1):
        _add_conv_block(model, filters)
        filters *= 2

    model.add(Flatten())

    units = 128
    for _ in range(num_dense_layers):
        _add_dense_block(model, units)
        units //= 2

    model.add(Dense(4, activation='softmax', kernel_regularizer=l2(0.01)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

    return model

def build_and_compile_model(architecture, image_size=224, dropout_rate=0.4, learning_rate=0.0001, regularizer_value=0.01, metrics=['accuracy', 'AUC']) -> Model:
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

def train_model_with_kfold(model, model_name, X_train, y_train, image_gen, n_splits=5) -> List:
    kf = KFold(n_splits=n_splits)
    fold = 0
    metrics_list = []
    
    for train_index, val_index in kf.split(X_train):
        fold += 1
        print(f"Training {model_name} on Fold {fold}")

        train_X, val_X = X_train[train_index], X_train[val_index]
        train_y, val_y = y_train[train_index], y_train[val_index]

        history = _train_model(model_name, image_gen, train_X, train_y, val_X, val_y, model)

        loss, acc, auc = model.evaluate(val_X, val_y)
        metrics_list.append([loss, acc, auc])
    
    avg_metrics = np.mean(metrics_list, axis=0)
    print(f"Averaged metrics for {model_name}: Loss: {avg_metrics[0]}, Accuracy: {avg_metrics[1]}, AUC: {avg_metrics[2]}")
    
    return metrics_list
    
def train_models_with_kfold(models, X_train, y_train, image_gen, n_splits=5) -> Dict:
    all_metrics = {}
    for model_name, model in models.items():
        metrics_list = train_model_with_kfold(model, model_name, X_train, y_train, image_gen, n_splits)
        all_metrics[model_name] = metrics_list
    return all_metrics
        
def display_summaries(models) -> None:
    for model in models.values():
        model.summary()
