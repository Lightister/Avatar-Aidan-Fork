import torch
import math

class DecisionTree:
    def __init__(self, max_depth=12, nb_features=None, is_categorical=None, n_bins=32, min_samples_split=50):
        self.max_depth = max_depth
        self.nb_features = nb_features
        self.is_categorical = is_categorical
        self.n_bins = n_bins
        self.min_samples_split = min_samples_split
        self.tree = None

    def _bin_features(self, X):
        X_binned = X.clone()
        numeric_cols = [i for i in range(X.shape[1]) if not self.is_categorical[i]]
        if not numeric_cols:
            return X_binned.long()

        for i in numeric_cols:
            col = X[:, i]
            edges = torch.linspace(col.min(), col.max(), self.n_bins, device=X.device)
            X_binned[:, i] = torch.searchsorted(edges, col.contiguous(), right=False)

        return X_binned.long()

    def fit(self, X, y):
        X_binned = self._bin_features(X)
        self.n_classes = int(y.max().item() + 1)
        self.nb_features = self.nb_features or int(math.sqrt(X.shape[1]))
        self.tree = self._build_tree(X_binned, y, depth=0)

    def _build_tree(self, X, y, depth):
        node = {}
        counts = torch.bincount(y, minlength=self.n_classes)
        pred_class = torch.argmax(counts).item()
        node['prediction'] = pred_class

        if depth >= self.max_depth or len(torch.unique(y)) == 1 or X.shape[0] < self.min_samples_split:
            node['is_leaf'] = True
            return node

        feature_idx = torch.randperm(X.shape[1])[:self.nb_features]
        best_gini = float('inf')
        best_feature = None
        best_split = None

        for f in feature_idx:
            vals = X[:, f]
            unique_vals = torch.unique(vals)
            for val in unique_vals[:-1]:
                left_mask = vals <= val
                right_mask = vals > val
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                gini = self._gini_impurity(y_left, y_right)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = f
                    best_split = val

        if best_feature is None:
            node['is_leaf'] = True
            return node

        node['is_leaf'] = False
        node['feature'] = best_feature
        node['split'] = best_split

        left_mask = X[:, best_feature] <= best_split
        right_mask = X[:, best_feature] > best_split

        node['left'] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _gini_impurity(self, y_left, y_right):
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        def gini(y):
            counts = torch.bincount(y, minlength=self.n_classes).float()
            probs = counts / counts.sum()
            return 1.0 - (probs ** 2).sum()
        return (n_left / n_total) * gini(y_left) + (n_right / n_total) * gini(y_right)

    def predict(self, X):
        X_binned = self._bin_features(X)
        n_samples = X.shape[0]
        preds = torch.full((n_samples,), -1, dtype=torch.long, device=X.device)

        queue = [(self.tree, torch.ones(n_samples, dtype=torch.bool, device=X.device))]

        while queue:
            node, mask = queue.pop()
            if mask.sum() == 0:
                continue
            if node['is_leaf']:
                preds[mask] = node['prediction']
                continue
            f, s = node['feature'], node['split']
            left_mask = mask & (X_binned[:, f] <= s)
            right_mask = mask & (X_binned[:, f] > s)
            queue.append((node['left'], left_mask))
            queue.append((node['right'], right_mask))

        return preds

class RandomForest:
    def __init__(self, 
                 n_estimators=100,
                 max_depth=12,
                 nb_features=None,
                 is_categorical=None,
                 columns=None,
                 bootstrap=True,
                 device='cpu',
                 sample_frac=0.1,
                 n_bins=32,
                 min_samples_split=50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.nb_features = nb_features
        self.is_categorical = is_categorical
        self.bootstrap = bootstrap
        self.device = device
        self.sample_frac = sample_frac
        self.n_bins = n_bins
        self.min_samples_split = min_samples_split
        self.trees = []

    def _train_single_tree(self, X, y):
        n_samples = max(1, int(X.shape[0] * self.sample_frac))
        idx = torch.randint(0, X.shape[0], (n_samples,), device=X.device)
        X_sub, y_sub = X[idx], y[idx]
        tree = DecisionTree(
            max_depth=self.max_depth,
            nb_features=self.nb_features,
            is_categorical=self.is_categorical,
            n_bins=self.n_bins,
            min_samples_split=self.min_samples_split
        )
        tree.fit(X_sub, y_sub)
        return tree

    def fit(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(self._train_single_tree(X, y))
        return self

    def predict(self, X):
        X = X.to(self.device)
        preds_list = [tree.predict(X) for tree in self.trees]
        preds_tensor = torch.stack(preds_list, dim=0)
        return torch.mode(preds_tensor, dim=0).values

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self, path)
        print(f"✅ Model saved to {path}")

    @staticmethod
    def load(path):
        # Import the classes that need to be available for unpickling
        from rf_model import RandomForest, DecisionTree
        # Temporarily make the classes available in __main__ for unpickling
        import sys
        main_module = sys.modules['__main__']
        main_module.RandomForest = RandomForest
        main_module.DecisionTree = DecisionTree
        try:
            with torch.serialization.safe_globals([RandomForest, DecisionTree]):
                if str(path).endswith('.gz'):
                    import gzip
                    with gzip.open(path, 'rb') as f:
                        model = torch.load(f, map_location='cpu', weights_only=False)
                else:
                    model = torch.load(path, map_location='cpu', weights_only=False)
        finally:
            # Clean up
            if hasattr(main_module, 'RandomForest'):
                delattr(main_module, 'RandomForest')
            if hasattr(main_module, 'DecisionTree'):
                delattr(main_module, 'DecisionTree')
        print(f"✅ Model loaded from {path}")
        return model