# Datasets, Generalization, and Overfitting - Simple ML Example

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

# Data characteristics
X, y = make_classification(n_samples=1000, n_features=20, weights=[0.9, 0.1], random_state=42)

# Imbalanced datasets - Upsample minority class
X_minority = X[y == 1]
y_minority = y[y == 1]
X_majority = X[y == 0]
y_majority = y[y == 0]

X_minority_upsampled, y_minority_upsampled = resample(
    X_minority, y_minority,
    replace=True,
    n_samples=len(y_majority),
    random_state=42
)

X_balanced = np.vstack((X_majority, X_minority_upsampled))
y_balanced = np.hstack((y_majority, y_minority_upsampled))

# Divide and transform data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model complexity, overfitting, L2 regularization
model = LogisticRegression(C=1.0, penalty='l2')  # C is inverse of regularization strength
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
