import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def generate_single_report(arch, metrics_list, model, X_test, y_test) -> None:
    avg_metrics = np.mean(metrics_list, axis=0)
    print(f"Averaged metrics for {arch}: Loss: {avg_metrics[0]}, Accuracy: {avg_metrics[1]}, AUC: {avg_metrics[2]}")
    
    predicted_classes = np.argmax(model.predict(X_test), axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.title(f'Confusion Matrix for {arch}')
    plt.show()
    
    print(f"Classification Report for {arch}:")
    print(classification_report(true_classes, predicted_classes))

def generate_reports(all_metrics, X_test, y_test, models) -> None:
    for arch, metrics_list in all_metrics.items():
        generate_single_report(arch, metrics_list, models[arch], X_test, y_test)

def evaluate_models(models, X_test, y_test) -> None:
    for arch, model in models.items():
        _evaluate_model(model, X_test, y_test, arch)

def _evaluate_model(model, X_test, y_test, arch) -> None:
    predicted_classes = np.argmax(model.predict(X_test), axis=1)
    confusionmatrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
    plt.show()

    print(classification_report(np.argmax(y_test, axis=1), predicted_classes))
