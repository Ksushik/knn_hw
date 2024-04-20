from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Compute Euclidean distance between two points.

    Args:
        point1 (np.ndarray): First point.
        point2 (np.ndarray): Second point.

    Returns:
        float: Euclidean distance between two points.
    """
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


class KNN:
    """K Nearest Neighbors classifier."""

    def __init__(self, k: int) -> None:
        """Initialize KNN with the number of neighbors to consider (k).

        Args:
            k (int): Number of neighbors to consider.
        """
        self._X_train = None
        self._y_train = None
        self.k = k  # number of neighbors to consider

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the KNN model with training data.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target.
        """
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Predict target values for test data.

        Args:
            X_test (np.ndarray): Test data features.
            verbose (bool, optional): Print progress during prediction. Defaults to False.

        Returns:
            np.ndarray: Predicted target values.
        """
        n = X_test.shape[0]
        y_pred = np.empty(n, dtype=self._y_train.dtype)

        for i in range(n):
            distances = np.array([euclidean_distance(x, X_test[i]) for x in self._X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self._y_train[k_indices]
            y_pred[i] = np.bincount(k_nearest_labels).argmax()

            if verbose:
                print(f"Predicted {i+1}/{n} samples", end="\r")

        if verbose:
            print("")
        return y_pred


def kfold_cross_validation(X: np.ndarray, y: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split dataset into k folds for cross-validation.

    Args:
        X (np.ndarray): Dataset features.
        y (np.ndarray): Dataset target.
        k (int): Number of folds.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: List of tuples (X_train, y_train, X_test, y_test).
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k
    folds = []  # Container to store the results of each fold

    n_samples = X.shape[0]
    fold_size = n_samples // k
    folds = []  # Container to store the results of each fold

    # Create indices for shuffling and splitting the data into folds
    indices = np.random.permutation(n_samples)
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else n_samples
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate((indices[:start_idx], indices[end_idx:]))

        # Split data into train and test sets for this fold
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        folds.append((X_train, y_train, X_test, y_test))

    return folds


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Accuracy score.
    """
    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # Calculate accuracy as the ratio of correct predictions to total predictions
    accuracy = correct_predictions / len(y_true)
    return accuracy


def main() -> None:
    """Main function to demonstrate the KNN classifier and k-fold cross-validation."""
    # Read training and testing data from CSV files
    # NOTE: data path, note that it must be specified relative to the \
    # directory from which you run this Python script
    training_data = pd.read_csv("data/train.csv")[:1000]
    testing_data = pd.read_csv("data/test.csv")

    # Extract features and target from the training data
    X = training_data.iloc[:, 1:].values
    y = training_data.iloc[:, 0].values
    print("Training data:", X.shape, y.shape)

    # Extract features and target from the testing data
    X_test = testing_data.iloc[:, 1:].values
    y_test = testing_data.iloc[:, 0].values
    print("Test data:", X_test.shape, y_test.shape)

    k = 1  # NOTE: not the best choice for k
    print(f" KNN with k = {k}")

    num_folds = 5
    # Perform k-fold cross-validation
    for X_train, y_train, X_val, y_val in kfold_cross_validation(X, y, k=num_folds):
        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val, verbose=True)
        accuracy = evaluate_accuracy(y_val, y_pred)
        print(f"Accuracy: {round(accuracy, 2)}")

    # Task 2 from README:
    # K from table to fill in from README
    test_k = [3, 4, 5, 6, 7, 9, 10, 15, 20, 21, 40, 41]
    # Here will store all accuracies to build graph later
    accuracies = []
    # Here will store max accuracy
    max_accuracy = 0
    # best k is k with maximum accuracy
    best_k = 0
    for n in range(1, 42):
        # Compute accuracy on test data
        model = KNN(k=n)
        model.fit(X, y)
        y_pred_test = model.predict(X_test)
        test_accuracy = evaluate_accuracy(y_test, y_pred_test)
        accuracies.append(test_accuracy)
        # compare accuracy with maximum accuracy so far
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            best_k = n
        # print accuracies needed to be filled in to table from README
        if n in test_k:
            print(f"Test accuracy for k={n}: {test_accuracy}")



    # Task 3 from README:

    cross_validation_accuracies = []
    # Perform k-fold cross-validation for best K
    for X_train, y_train, X_val, y_val in kfold_cross_validation(X, y, k=num_folds):
        model = KNN(k=best_k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = evaluate_accuracy(y_val, y_pred)
        cross_validation_accuracies.append(round(accuracy, 2))
    avg_cv_accuracy = np.mean(cross_validation_accuracies)
    print(f"Avg accuracy for k={best_k}: {avg_cv_accuracy:.2f}")



    k_list = range(1, 21)
    plt.plot(k_list, accuracies[1:21])

    # Highlight best accuracy and k for the highest accuracy
    plt.scatter(best_k, max_accuracy, color='red', label=f'Max Accuracy: {max_accuracy} at k={best_k}')
    plt.text(best_k, max_accuracy, f'best_k={best_k}', ha='right', va='bottom')

    # Add a horizontal line for avg_cv_accuracy
    plt.axhline(y=avg_cv_accuracy, color='green', linestyle='--', label=f'Average CV Accuracy for k={best_k}: {avg_cv_accuracy:.3f}')
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("KNN Model Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
