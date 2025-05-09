import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
        

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature, best_threshold = None, None

        parent_impurity = self.gini_impurity(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue

                left_y, right_y = y[left_mask], y[right_mask]
                left_impurity = self.gini_impurity(left_y)
                right_impurity = self.gini_impurity(right_y)

                weighted_impurity = (len(left_y) / n_samples) * left_impurity + (len(right_y) / n_samples) * right_impurity
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def fit(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            return {"leaf": True, "class": np.bincount(y).argmax()}

        # Find best split
        feature, threshold, gain = self.find_best_split(X, y)
        if feature is None or gain <= 0:
            return {"leaf": True, "class": np.bincount(y).argmax()}

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # Recursively build tree
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1)
        }

    def predict(self, X):
        self.tree = self.fit(X, y)  # Fit tree if not already fitted
        predictions = []
        for x in X:
            node = self.tree
            while not node["leaf"]:
                if x[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node["class"])
        return np.array(predictions)

# Example usage
X = np.array([[1, 2], [2, 3], [3, 1], [4, 4]])
y = np.array([1, 0, 1, 1])
dt = DecisionTree(max_depth=3)
predictions = dt.predict(X)
print(predictions)