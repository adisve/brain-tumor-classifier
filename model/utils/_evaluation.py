import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def generate_single_report(model_name, metrics_list, model, X_test, y_test):
    avg_metrics = np.mean(metrics_list, axis=0)
    print(f"Averaged metrics for {model_name}: Loss: {avg_metrics[0]}, Accuracy: {avg_metrics[1]}, AUC: {avg_metrics[2]}")
    
    predicted_classes = np.argmax(model.predict(X_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()
    
    print(f"Classification Report for {model_name}:")
    print(classification_report(true_classes, predicted_classes))

def generate_reports(all_metrics, X_test, y_test, models):
    for model_name, metrics_list in all_metrics.items():
        generate_single_report(model_name, metrics_list, models[model_name], X_test, y_test)

def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        __evaluate_model(model, X_test, y_test, model_name)

def __evaluate_model(model, X_test, y_test, model_name):
    predicted_classes = np.argmax(model.predict(X_test), axis=1)
    confusionmatrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(np.argmax(y_test, axis=1), predicted_classes))
