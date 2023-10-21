import matplotlib.pyplot as plt

def plot(metric, history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history[metric], 'r', linewidth=3.0)
    plt.plot(history.history[f'val_{metric}'], 'b', linewidth=3.0)
    plt.legend([f'Training {metric}', f'Validation {metric}'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title(f'{metric} Curves', fontsize=16)
    plt.show()
