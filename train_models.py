import pandas as pd
import numpy as np
from cvxopt import matrix, solvers

import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Custom SVM implementation
class CustomSVM:
    def __init__(self, C=1.0, gamma='scale', learning_rate=0.01, max_iterations=1000):
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.support_vectors_ = None
        self.X_train = None
        self.y_train = None
        self.support_vector_weights_ = None
        self.support_vector_labels_ = None

    def rbf_kernel(self, X1, X2):
        # Convert pandas DataFrame/Series to NumPy array if necessary
        if isinstance(X1, (pd.DataFrame, pd.Series)):
            X1 = X1.to_numpy()
        if isinstance(X2, (pd.DataFrame, pd.Series)):
            X2 = X2.to_numpy()

        # Tính kernel RBF: exp(-gamma * ||x1 - x2||^2)
        gamma = self.gamma
        if gamma == 'scale':
            gamma = 1.0 / (X1.shape[1] * np.var(X1))
        
        # Ensure X1 and X2 are 2D arrays
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        squared_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * squared_dist)

    def fit(self, X, y):
        # Convert pandas DataFrame/Series to NumPy array
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        n_samples, n_features = X.shape
        # Chuyển nhãn thành {-1, 1}
        y_ = np.where(y == 0, -1, 1)
        
        # Khởi tạo weights và bias
        self.weights = np.zeros(n_samples)  # Dual coefficients (alpha)
        self.bias = 0.0
        
        # Lưu dữ liệu để tính kernel
        self.X_train = X
        self.y_train = y_
        
        # Gradient descent để tối ưu hóa
        for _ in range(self.max_iterations):
            # Tính dự đoán
            K = self.rbf_kernel(X, X)
            y_pred = np.sign(np.dot(K, self.weights * y_) + self.bias)
            
            # Cập nhật weights (alpha)
            for i in range(n_samples):
                if y_[i] * (np.dot(K[i], self.weights * y_) + self.bias) <= 1:
                    self.weights[i] += self.learning_rate * (self.C * y_[i] - (self.weights[i] * y_[i] * K[i, i]))
                else:
                    self.weights[i] -= self.learning_rate * self.weights[i]
            
            # Cập nhật bias
            self.bias += self.learning_rate * np.mean(y_ - np.dot(K, self.weights * y_))
        
        # Lưu support vectors
        sv_indices = np.abs(self.weights) > 1e-5
        self.support_vectors_ = X[sv_indices]
        self.support_vector_labels_ = y_[sv_indices]
        self.support_vector_weights_ = self.weights[sv_indices]


    def predict(self, X):
        # Convert pandas DataFrame/Series to NumPy array
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()

        # Compute kernel between test data and support vectors (not all training data)
        K = self.rbf_kernel(X, self.support_vectors_)
        decision = np.dot(K, self.support_vector_weights_ * self.support_vector_labels_) + self.bias
        return np.where(np.sign(decision) == 1, 1, 0)

# Read data
data = pd.read_csv('features_handcraft.csv')
X = data.drop('label', axis=1)
y = data['label']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA (512 components)
pca = PCA(n_components=512)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train SVC
svm = SVC(probability=True, kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Report:\n", classification_report(y_test, svm_pred))

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200,max_depth=20 ,random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Report:\n", classification_report(y_test, rf_pred))

# Train Custom SVM
custom_svm = CustomSVM(C=1.0, gamma='scale', learning_rate=0.01, max_iterations=1000)
custom_svm.fit(X_train, y_train)
custom_svm_pred = custom_svm.predict(X_test)
print("Custom SVM Accuracy:", accuracy_score(y_test, custom_svm_pred))
print("Custom SVM Report:\n", classification_report(y_test, custom_svm_pred))

# Save models
joblib.dump(scaler, 'models/scaler_model.pkl')
joblib.dump(pca, 'models/pca_model.pkl')
joblib.dump(svm, 'models/svm_model.pkl')
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(custom_svm, 'custom_svm_model.pkl')
print("Saved files: scaler_model.pkl, pca_model.pkl, svm_model.pkl, rf_model.pkl, custom_svm_model.pkl")