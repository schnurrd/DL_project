from sklearn.manifold import TSNE
from sklearn import metrics
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from globals import SEED, CLASSES_PER_ITER

colors = [
        'lightblue', 'darkblue',   # 0 and 1
        'lightgreen', 'darkgreen', # 2 and 3
        'orange', 'gold',     # 4 and 5
        'lightcoral', 'darkred',   # 6 and 7
        'plum', 'purple'           # 8 and 9
    ]

def plot_embeddings(embeddings, labels, num_classes, centers_tens=None):
    custom_colors = colors[:num_classes]
    cmap = ListedColormap(custom_colors)
    
    # Concatenate embeddings and centers for a consistent transformation
    centers = []
    if centers_tens is not None:
        for (i, c) in enumerate(centers_tens):
            centers.append(centers_tens[i].detach().cpu().numpy())
        centers = np.array(centers)
        assert len(centers) == num_classes, "The number of centers must match the number of classes. Centers length " + str(len(centers)) + " num classes " + str(num_classes)
        all_data = np.vstack([embeddings, centers])
    else:
        all_data = embeddings

    # Apply TSNE to the combined data
    tsne = TSNE(n_components=2, random_state=SEED)
    all_data_2d = tsne.fit_transform(all_data)
    
    # Separate transformed embeddings and centers
    embeddings_2d = all_data_2d[:len(embeddings)]
    if centers is not None:
        centers_2d = all_data_2d[len(embeddings):]
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot for embeddings
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, alpha=0.6)
    
    # Plot centers if provided
    if centers_tens is not None:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c=range(num_classes), cmap=cmap, 
                    marker='X', s=200, edgecolor='black', label="Centers")
    
    # Title and legend
    text = "2D Visualization of Class Embeddings"
    if centers_tens is not None:
        text += " and Centers"
    plt.title(text)
    legend_handles = [mpatches.Patch(color=custom_colors[i], label=f'Class {i}') for i in range(num_classes)]
    plt.legend(handles=legend_handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_confusion_matrix(pred, true, display_labels):
    # Check if inputs are tensors, and convert them to numpy arrays if necessary
    if hasattr(true, "cpu"):
        true = true.cpu().numpy()
    elif isinstance(true, np.ndarray):
        true = true
    else:
        true = np.array(true)

    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()
    elif isinstance(pred, np.ndarray):
        pred = pred
    else:
        pred = np.array(pred)

    # Compute confusion matrix
    confusion_matrix = metrics.confusion_matrix(true, pred)

    # Display confusion matrix
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    cm_display.plot()
    plt.show()