# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
import time

# Reproducibility
np.random.seed(42)


def load_and_preprocess_data(file_path='heart.csv'):
    """Load and preprocess heart disease dataset with cleaning, feature engineering, and outlier handling."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    # Clean problematic values and convert to numeric
    df.replace(['?', '', 'twenty', 'ss'], np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'target' not in df.columns:
        print("Error: 'target' column not found.")
        return None

    # Ensure target is properly formatted
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)

    # Handle missing values
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 0)

    # Cap outliers using IQR method
    def cap_outliers(series):
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        bounds = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return series.clip(*bounds)

    for feature in num_cols:
        if feature in df.columns:
            df[feature] = cap_outliers(df[feature])

    # Feature engineering
    epsilon = 1e-6
    df['bp_hr_ratio'] = df['trestbps'] / (df['thalach'] + epsilon)
    df['chol_age_ratio'] = df['chol'] / (df['age'] + epsilon)

    # Create age and cholesterol categories
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 40, 55, 65, df['age'].max() + 1],
                             labels=['young', 'middle_aged', 'senior', 'elderly'],
                             right=False)

    df['chol_level'] = pd.cut(df['chol'],
                              bins=[0, 200, 240, df['chol'].max() + 1],
                              labels=['normal', 'borderline_high', 'high'],
                              right=False)

    # One-hot encode categorical features
    cat_to_encode = ['cp', 'restecg', 'slope', 'ca', 'thal', 'age_group', 'chol_level']
    existing_cats = [col for col in cat_to_encode if col in df.columns]

    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # Final data quality checks
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicates.")

    if df.isnull().sum().sum() > 0:
        print("Warning: Remaining NaNs after preprocessing:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    print("Preprocessing complete.")
    return df


def select_features(X, y, k=10):
    """Select top k features using ANOVA F-test."""
    k = min(k, X.shape[1])
    if k <= 0:
        print("Warning: Invalid k value. Using all features.")
        return X.columns.tolist()

    print(f"\nSelecting top {k} features...")
    selector = SelectKBest(score_func=f_classif, k=k)

    try:
        selector.fit(X, y)
        selected = X.columns[selector.get_support()]
        print("\nSelected Features:", selected.tolist())
        return selected.tolist()
    except Exception as e:
        print(f"Feature selection error: {e}\nUsing all features.")
        return X.columns.tolist()


def visualize_data(df, target_col='target'):
    """Generate exploratory visualizations including correlation matrix and target distribution."""
    if df is None or df.empty:
        print("No data to visualize.")
        return

    print("\nGenerating visualizations...")

    # Correlation heatmap
    plt.figure(figsize=(18, 15))
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, mask=mask, cmap='Spectral', vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=True, fmt='.2f', annot_kws={"size": 8})

        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Target distribution
    if target_col in df.columns:
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x=target_col, data=df, palette='Set2', hue=target_col)
        plt.title('Target Variable Distribution')

        # Add percentage labels
        total = len(df[target_col])
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                        ha='center', fontsize=10)
        plt.show()

    # Age distribution by target
    if 'age' in df.columns and target_col in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='age', hue=target_col, kde=True,
                     bins=20, palette='Set2', alpha=0.6)
        plt.title('Age Distribution by Heart Disease Status')
        plt.legend(title='Heart Disease', labels=['No', 'Yes'])
        plt.show()

    print("Visualizations complete.")


class KNNFromScratch:
    """KNN implementation from scratch with basic functionality."""

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """Predict labels for test data."""
        X_test = np.array(X)
        if self.X_train is None:
            raise ValueError("Model not fitted yet.")
        return np.array([self._predict_single(x) for x in X_test])

    def _predict_single(self, x):
        """Helper method to predict single sample."""
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_nearest = self.y_train[np.argsort(distances)[:self.k]]
        return Counter(k_nearest).most_common(1)[0][0]

    def score(self, X, y):
        """Calculate accuracy score."""
        return np.mean(self.predict(X) == np.array(y))


def main():
    """Main execution pipeline: data loading, preprocessing, modeling, and evaluation."""
    # Data loading and preparation
    df = load_and_preprocess_data()
    if df is None or 'target' not in df.columns:
        print("Aborting: Invalid data.")
        return

    visualize_data(df)

    # Feature selection
    X = df.drop('target', axis=1)
    y = df['target']
    features = select_features(X, y, k=15)
    X = X[features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model configuration
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs']
        }),
        "SVM": (SVC(random_state=42, probability=True), {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        }),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5]
        }),
        "KNN (sklearn)": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        })
    }

    # Model training and evaluation
    performance = {}
    best_params = {}

    for name, (model, params) in models.items():
        print(f"\n--- Training {name} ---")
        start = time.time()

        grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        best_params[name] = grid.best_params_
        y_pred = best_model.predict(X_test_scaled)

        performance[name] = accuracy_score(y_test, y_pred)

        print(f"Best params: {best_params[name]}")
        print(f"Accuracy: {performance[name]:.4f}")
        print(f"Time: {time.time() - start:.2f}s")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Custom KNN evaluation
    best_k = best_params.get("KNN (sklearn)", {}).get("n_neighbors", 7)
    print(f"\n--- Evaluating KNN (scratch) with k={best_k} ---")

    knn = KNNFromScratch(k=best_k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    performance['KNN (scratch)'] = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {performance['KNN (scratch)']:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Results comparison
    print("\n--- Final Model Performance ---")
    sorted_perf = dict(sorted(performance.items(), key=lambda x: x[1], reverse=True))

    for model, acc in sorted_perf.items():
        print(f"{model}: {acc:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_perf.keys(), sorted_perf.values())
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}',
                 va='bottom', ha='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()