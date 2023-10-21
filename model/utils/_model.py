from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNetV2, ResNet50
from utils._callbacks import init_callbacks, init_earlystopping_and_scheduler
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

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

def train_model_with_kfold(model, model_name, X_train, y_train, image_gen, n_splits=5) -> list:
    kf = KFold(n_splits=n_splits)
    fold = 0
    metrics_list = []
    
    for train_index, val_index in kf.split(X_train):
        fold += 1
        print(f"Training {model_name} on Fold {fold}")

        train_X, val_X = X_train[train_index], X_train[val_index]
        train_y, val_y = y_train[train_index], y_train[val_index]

        history = train_model(model_name, image_gen, train_X, train_y, val_X, val_y, model)

        loss, acc, auc = model.evaluate(val_X, val_y)
        metrics_list.append([loss, acc, auc])
    
    avg_metrics = np.mean(metrics_list, axis=0)
    print(f"Averaged metrics for {model_name}: Loss: {avg_metrics[0]}, Accuracy: {avg_metrics[1]}, AUC: {avg_metrics[2]}")
    
    return metrics_list
    
def train_models_with_kfold(models, X_train, y_train, image_gen, n_splits=5) -> dict:
    all_metrics = {}
    for model_name, model in models.items():
        metrics_list = train_model_with_kfold(model, model_name, X_train, y_train, image_gen, n_splits)
        all_metrics[model_name] = metrics_list
    return all_metrics
        
def display_summaries(models) -> None:
    for model in models.values():
        model.summary()

def train_model(model_name, image_gen, X_train, y_train, X_val, y_val, model, batch_size=64, epochs=30, min_delta=0.001, patience=3, lr_factor=0.3):
    early_stopping, scheduler = init_earlystopping_and_scheduler(min_delta, patience, lr_factor)
    callbacks = init_callbacks(model_name, early_stopping, scheduler)
    model.fit(
        image_gen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )
    return model
