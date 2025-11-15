import numpy as np
from collections import Counter
from numpy.linalg import eigh
import pandas as pd

class PCAModel:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


class KNNClassifier:
    def __init__(self, n_neighbors = 5, metric = 'euclidean', weights = 'distance'):
            
        self.n_neighbors = n_neighbors
        self.metric = metric.lower()
        self.weights = weights.lower()
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def calculate_distances(self, X_test: np.ndarray):
        
        if self.metric == 'euclidean':
            X_test_sq = np.sum(X_test**2, axis=1, keepdims=True)
            X_train_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T
            dot_prod = np.dot(X_test, self.X_train.T)
            
            dists_sq = X_test_sq - 2 * dot_prod + X_train_sq
            dists_sq = np.maximum(dists_sq, 0)
            return np.sqrt(dists_sq)

        elif self.metric == 'manhattan':
            diff = self.X_train[np.newaxis, :, :] - X_test[:, np.newaxis, :]
            return np.sum(np.abs(diff), axis=2)
        
        return None

    def fit(self, X, y):
        
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values.ravel()
            
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        
        return self

    def predict(self, X, batch_size = 512):

        X = np.asarray(X, dtype=np.float32)
        if self.X_train is None: raise ValueError("Model not fitted.")
        
        n_test = X.shape[0]
        predictions = np.empty(n_test, dtype=self.y_train.dtype)
        
        for start in range(0, n_test, batch_size):
            end = min(n_test, start + batch_size)
            X_chunk = X[start:end]
            
            dists = self.calculate_distances(X_chunk)
            
            n_chunk = X_chunk.shape[0]
            k = self.n_neighbors
            n_train = dists.shape[1]
            k = min(k, n_train)
            
            idx_part = np.argpartition(dists, kth=k-1, axis=1)[:, :k] 
            rows = np.arange(n_chunk)[:, None]
            
            k_dists = dists[rows, idx_part]
            k_labels = self.y_train[idx_part]
            
            order = np.argsort(k_dists, axis=1)
            k_labels = np.take_along_axis(k_labels, order, axis=1)
            k_dists = np.take_along_axis(k_dists, order, axis=1)

            if self.weights == 'uniform':
                for i in range(n_chunk):
                    predictions[start + i] = Counter(k_labels[i]).most_common(1)[0][0]
            else:
                epsilon = 1e-12
                weights_arr = 1.0 / (k_dists + epsilon)
                weights_arr[k_dists < epsilon] = 1.0

                for i in range(n_chunk):
                    weighted_votes = {}
                    for label, weight in zip(k_labels[i], weights_arr[i]):
                        weighted_votes[label] = weighted_votes.get(label, 0) + weight
                    predictions[start + i] = max(weighted_votes, key=weighted_votes.get)

        return predictions
    
    
class Ensemble:
    def __init__(self, knn_cls = KNNClassifier, pca_cls = PCAModel, n_estimators = 10, ensemble_type = 'bagging',
                 sample_fraction = 1.0, random_state = None, pca_n_components = None, **knn_kwargs):

        self.knn_cls = knn_cls
        self.pca_cls = pca_cls
        self.n_estimators = n_estimators
        self.ensemble_type = ensemble_type.lower()
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.pca_n_components = pca_n_components
        self.knn_kwargs = knn_kwargs

        self.pca_model = None
        self.models = []       
        self.model_weights = []
        self.classes_ = None
        
        self._rng = np.random.default_rng(self.random_state)
        
    def handle_input(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values.ravel()
        return np.asarray(X, dtype=np.float32), np.asarray(y)

    def fit_and_transform_pca(self, X):
        if self.pca_n_components is None or self.pca_n_components >= X.shape[1]:
            self.pca_model = None
            return X
    
        self.pca_model = self.pca_cls(n_components=self.pca_n_components)
        self.pca_model.fit(X)
        return self.pca_model.transform(X)

    def transform_pca(self, X):
        if self.pca_model is None:
            return np.asarray(X, dtype=np.float32)
        return self.pca_model.transform(X)

    def fit(self, X, y):
        X, y = self.handle_input(X, y)
        self.classes_ = np.unique(y)
        self.models = []
        self.model_weights = []

        X_proc = self.fit_and_transform_pca(X)
        
        if self.ensemble_type == 'bagging':
            self.fit_bagging(X_proc, y)
        return self

    def fit_bagging(self, X, y):
        n_samples = X.shape[0]
        m = max(1, int(np.round(self.sample_fraction * n_samples))) 

        for i in range(self.n_estimators):
            indices = self._rng.choice(n_samples, size=m, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            clf = self.knn_cls(**self.knn_kwargs)
            clf.fit(X_boot, y_boot)
            self.models.append(clf)
            self.model_weights.append(1.0)

    def predict(self, X):
        X_proc = self.transform_pca(X)
        
        n_samples = X_proc.shape[0]
        votes = np.zeros((n_samples, len(self.models)), dtype=self.classes_.dtype)
        
        for i, clf in enumerate(self.models):
            votes[:, i] = clf.predict(X_proc)

        final_predictions = np.empty(n_samples, dtype=self.classes_.dtype)
        
        for j in range(n_samples):
            if self.ensemble_type == 'bagging':
                c = Counter(votes[j])
                final_predictions[j] = c.most_common(1)[0][0]
        return final_predictions



    
    
