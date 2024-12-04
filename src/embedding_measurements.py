import numpy as np
from sklearn.neighbors import NearestNeighbors
import globals
import copy

def measure_embedding_confusion_knn(embeddings, labels, k=10, task=globals.ITERATIONS):
    try:
        labels = labels.cpu().numpy()
    except:
        pass
    try:
        embeddings = embeddings.cpu().numpy()
    except:
        pass
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    indices = indices[:, 1:]  # Exclude self-neighbor
    labels_of_neighbors = labels[indices]

    # First, calculate total confusion (proportion of KNN embeddings from ANY other task and class)
    same_class = (labels_of_neighbors == labels[:, None])
    total_confusion = 1 - same_class.mean()

    # Then, calculate intra-phase confusion (proportion of KNN embeddings from other TASKS)
    def same_task(label1, label2):
        return (label1 // globals.CLASSES_PER_ITER) == (label2 // globals.CLASSES_PER_ITER)
    
    result = []
    for l, ls_of_knn in zip(labels, labels_of_neighbors):
        result.append([same_task(l, l_of_knn) for l_of_knn in ls_of_knn])
    result = np.array(result)
    intra_phase_confusion = 1 - result.mean()

    per_task_confusions = []
    # Lastly, calculate within-task confusion ("total" confusion, but measured for every task separately)
    for i in range(task):
        task_labels = [l for l in range(i*globals.CLASSES_PER_ITER, (i+1)*globals.CLASSES_PER_ITER)]
        mask = np.isin(labels, task_labels)
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(filtered_embeddings)
        _, indices = nbrs.kneighbors(filtered_embeddings)
        indices = indices[:, 1:]  # Exclude self-neighbor
        labels_of_neighbors = filtered_labels[indices]
        same_task = (labels_of_neighbors == filtered_labels[:, None])
        task_confusion = 1 - same_task.mean()
        per_task_confusions.append(task_confusion)
    avg_per_task_confusion = sum(per_task_confusions)/len(per_task_confusions)

    return total_confusion, intra_phase_confusion, avg_per_task_confusion

def measure_embedding_drift(embeddings, labels, prev_embedding_centers):
    _prev_embedding_centers = copy.deepcopy(prev_embedding_centers)
    try:
        labels = labels.cpu().numpy()
    except:
        pass
    try:
        _prev_embedding_centers = _prev_embedding_centers.cpu().numpy()
    except:
        pass
    try:
        for(i, c) in enumerate(_prev_embedding_centers):
            _prev_embedding_centers[i] = c.cpu().numpy()
    except:
        pass
    try:
        embeddings = embeddings.cpu().numpy()
    except:
        pass

    labels = np.array(labels)
    _prev_embedding_centers = np.array(_prev_embedding_centers)
    embeddings = np.array(embeddings)
    classes = [i for i in range(globals.CLASSES_PER_ITER*globals.ITERATIONS)]
    drifts = []
    for (i, c) in enumerate(classes):
        class_embeddings = embeddings[labels == c]
        if class_embeddings.shape[0] == 0:
            continue
        center = np.mean(class_embeddings, axis=0)
        distance = np.linalg.norm(center - _prev_embedding_centers[i])
        drifts.append(distance)
    drifts = np.array(drifts)
    return float(drifts.mean())