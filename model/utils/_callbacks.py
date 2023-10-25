from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import List, Tuple

def init_callbacks(model_name, early_stopping, scheduler) -> List:
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

def init_earlystopping_and_scheduler(min_delta, patience, lr_factor) -> Tuple:
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
