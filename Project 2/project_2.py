# CIFAR-10 SVM Classification Project - EXTENDED EXPERIMENTS VERSION
# Neural Networks - Deep Learning Course
# Features: Multiple experiments + comprehensive logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from tensorflow import keras
import time
import warnings
import os
import joblib
import pandas as pd
from datetime import datetime
import sys
warnings.filterwarnings('ignore')

# ===== LOGGING SETUP =====
class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'experiment_log_{timestamp}.txt')
        self.results_file = os.path.join(log_dir, f'results_{timestamp}.csv')
        
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def log_separator(self, char='=', length=60):
        """Log a separator line"""
        self.log(char * length)
    
    def save_results_table(self, df, name="results"):
        """Save results dataframe"""
        csv_path = self.log_file.replace('.txt', f'_{name}.csv')
        df.to_csv(csv_path, index=False)
        self.log(f"Results table saved to: {csv_path}")

# ===== CONFIGURATION =====
save_dir = './CIFAR10_SVM_Experiments'
os.makedirs(save_dir, exist_ok=True)

logger = Logger(save_dir)
logger.log("="*60)
logger.log("CIFAR-10 SVM CLASSIFICATION - EXTENDED EXPERIMENTS")
logger.log("="*60)

# Experiment configuration
EXPERIMENTS = {
    'subset_sizes': [1000, 5000, 10000],  # Different dataset sizes
    'pca_variance': [0.85, 0.90, 0.95],   # Different PCA thresholds
    'linear_C': [0.1, 1.0, 10.0],          # Linear SVM C values
    'rbf_C': [1.0, 10.0, 100.0],           # RBF SVM C values
    'rbf_gamma': ['scale', 0.001, 0.01],   # RBF gamma values
    'knn_neighbors': [1, 3, 5, 7],         # KNN k values
    'mlp_hidden': [50, 100, 200],          # MLP hidden units
}

# Set which experiments to run
RUN_EXPERIMENTS = {
    'baseline': True,              # Standard run with default params
    'subset_comparison': True,     # Compare different dataset sizes
    'pca_comparison': True,        # Compare PCA variance thresholds
    'hyperparameter_tuning': True, # Test different C, gamma values
    'knn_comparison': False,        # Test different k values
    'binary_classification': True, # Test on 2-class problems
}

SUBSET_SIZE = None  # Main subset size (set to None for full dataset)

# ===== LOAD DATA =====
logger.log("\nLoading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

logger.log(f"Training set shape: {X_train.shape}")
logger.log(f"Test set shape: {X_test.shape}")

# Flatten and normalize
X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
y_train = y_train.ravel()
y_test = y_test.ravel()

# Standardize
logger.log("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# ===== EXPERIMENT 1: BASELINE WITH STANDARD PARAMETERS =====
if RUN_EXPERIMENTS['baseline']:
    logger.log_separator()
    logger.log("EXPERIMENT 1: BASELINE WITH STANDARD PARAMETERS")
    logger.log_separator()
    
    # PCA
    logger.log("\nApplying PCA (90% variance)...")
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    logger.log(f"Original dimensions: {X_train_scaled.shape[1]}")
    logger.log(f"PCA dimensions: {X_train_pca.shape[1]}")
    logger.log(f"Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Subset selection
    if SUBSET_SIZE:
        indices = np.random.choice(X_train_pca.shape[0], min(SUBSET_SIZE, X_train_pca.shape[0]), replace=False)
        X_train_subset = X_train_pca[indices]
        y_train_subset = y_train[indices]
        logger.log(f"\nUsing subset of {len(X_train_subset)} samples")
    else:
        X_train_subset = X_train_pca
        y_train_subset = y_train
        logger.log(f"\nUsing full dataset of {len(X_train_subset)} samples")
    
    results = []
    
    # Linear SVM
    logger.log("\n1. Training Linear SVM...")
    start = time.time()
    svm_linear = LinearSVC(C=1.0, max_iter=1000, random_state=42)
    svm_linear.fit(X_train_subset, y_train_subset)
    svm_linear_cal = CalibratedClassifierCV(svm_linear, cv=3)
    svm_linear_cal.fit(X_train_subset, y_train_subset)
    train_time = time.time() - start
    
    y_pred = svm_linear_cal.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
    results.append(['Linear SVM', train_time, acc])
    
    # RBF SVM
    logger.log("\n2. Training RBF SVM...")
    start = time.time()
    svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_train_subset, y_train_subset)
    train_time = time.time() - start
    
    y_pred_rbf = svm_rbf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred_rbf)
    logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
    results.append(['RBF SVM', train_time, acc])
    
    # Polynomial SVM
    logger.log("\n3. Training Polynomial SVM...")
    start = time.time()
    svm_poly = SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42)
    svm_poly.fit(X_train_subset, y_train_subset)
    train_time = time.time() - start
    
    y_pred = svm_poly.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
    results.append(['Poly SVM', train_time, acc])
    
    # KNN
    logger.log("\n4. Training 1-NN...")
    start = time.time()
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(X_train_subset, y_train_subset)
    train_time = time.time() - start
    
    y_pred = knn1.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
    results.append(['1-NN', train_time, acc])
    
    logger.log("\n5. Training 3-NN...")
    start = time.time()
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(X_train_subset, y_train_subset)
    train_time = time.time() - start
    
    y_pred = knn3.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
    results.append(['3-NN', train_time, acc])
    
    # Nearest Centroid
    logger.log("\n6. Training Nearest Centroid...")
    start = time.time()
    centroids = []
    for i in range(10):
        class_samples = X_train_subset[y_train_subset == i]
        centroid = np.mean(class_samples, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    def predict_centroid(X, centroids):
        predictions = []
        for x in X:
            distances = np.linalg.norm(centroids - x, axis=1)
            predictions.append(np.argmin(distances))
        return np.array(predictions)
    
    y_pred = predict_centroid(X_test_pca, centroids)
    train_time = time.time() - start
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
    results.append(['Nearest Centroid', train_time, acc])
    
    # MLP
    logger.log("\n7. Training MLP...")
    start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                        solver='sgd', max_iter=50, random_state=42)
    mlp.fit(X_train_subset, y_train_subset)
    train_time = time.time() - start
    
    y_pred = mlp.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
    results.append(['MLP', train_time, acc])
    
    # Save baseline results
    df_baseline = pd.DataFrame(results, columns=['Method', 'Train Time (s)', 'Test Accuracy'])
    logger.save_results_table(df_baseline, 'baseline')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_rbf)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - RBF SVM (Baseline)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp1_confusion_matrix.svg', format='svg', bbox_inches='tight')
    plt.close()
    logger.log("Confusion matrix saved")

# ===== EXPERIMENT 2: SUBSET SIZE COMPARISON =====
if RUN_EXPERIMENTS['subset_comparison']:
    logger.log_separator()
    logger.log("EXPERIMENT 2: SUBSET SIZE COMPARISON")
    logger.log_separator()
    
    # Use 90% PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    subset_results = []
    
    for subset_size in EXPERIMENTS['subset_sizes']:
        logger.log(f"\nTesting with {subset_size} samples...")
        
        indices = np.random.choice(X_train_pca.shape[0], subset_size, replace=False)
        X_sub = X_train_pca[indices]
        y_sub = y_train[indices]
        
        # Train Linear SVM
        start = time.time()
        svm = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        svm.fit(X_sub, y_sub)
        train_time = time.time() - start
        y_pred = svm.predict(X_test_pca)
        acc_linear = accuracy_score(y_test, y_pred)
        
        # Train RBF SVM
        start = time.time()
        svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        svm_rbf.fit(X_sub, y_sub)
        train_time_rbf = time.time() - start
        y_pred_rbf = svm_rbf.predict(X_test_pca)
        acc_rbf = accuracy_score(y_test, y_pred_rbf)
        
        logger.log(f"   Linear SVM - Time: {train_time:.2f}s, Acc: {acc_linear:.4f}")
        logger.log(f"   RBF SVM    - Time: {train_time_rbf:.2f}s, Acc: {acc_rbf:.4f}")
        
        subset_results.append([subset_size, train_time, acc_linear, train_time_rbf, acc_rbf])
    
    df_subset = pd.DataFrame(subset_results, 
                            columns=['Subset Size', 'Linear Time (s)', 'Linear Acc', 
                                   'RBF Time (s)', 'RBF Acc'])
    logger.save_results_table(df_subset, 'subset_comparison')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df_subset['Subset Size'], df_subset['Linear Acc'], 'o-', label='Linear SVM')
    ax1.plot(df_subset['Subset Size'], df_subset['RBF Acc'], 's-', label='RBF SVM')
    ax1.set_xlabel('Subset Size')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df_subset['Subset Size'], df_subset['Linear Time (s)'], 'o-', label='Linear SVM')
    ax2.plot(df_subset['Subset Size'], df_subset['RBF Time (s)'], 's-', label='RBF SVM')
    ax2.set_xlabel('Subset Size')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp2_subset_comparison.svg', format='svg', bbox_inches='tight')
    plt.close()
    logger.log("Subset comparison plots saved")

# ===== EXPERIMENT 3: PCA VARIANCE COMPARISON =====
if RUN_EXPERIMENTS['pca_comparison']:
    logger.log_separator()
    logger.log("EXPERIMENT 3: PCA VARIANCE THRESHOLD COMPARISON")
    logger.log_separator()
    
    if SUBSET_SIZE:
        indices = np.random.choice(X_train_scaled.shape[0], SUBSET_SIZE, replace=False)
        X_sub_scaled = X_train_scaled[indices]
        y_sub = y_train[indices]
    else:
        X_sub_scaled = X_train_scaled
        y_sub = y_train
    
    pca_results = []
    
    for variance in EXPERIMENTS['pca_variance']:
        logger.log(f"\nTesting with {variance*100:.0f}% variance...")
        
        pca = PCA(n_components=variance, random_state=42)
        X_pca = pca.fit_transform(X_sub_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        n_components = X_pca.shape[1]
        actual_variance = pca.explained_variance_ratio_.sum()
        
        # Train RBF SVM
        start = time.time()
        svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        svm.fit(X_pca, y_sub)
        train_time = time.time() - start
        
        y_pred = svm.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        
        logger.log(f"   Components: {n_components}, Actual variance: {actual_variance:.4f}")
        logger.log(f"   Time: {train_time:.2f}s, Accuracy: {acc:.4f}")
        
        pca_results.append([variance, n_components, actual_variance, train_time, acc])
    
    df_pca = pd.DataFrame(pca_results, 
                         columns=['Target Variance', 'Components', 'Actual Variance', 
                                'Time (s)', 'Accuracy'])
    logger.save_results_table(df_pca, 'pca_comparison')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df_pca['Components'], df_pca['Accuracy'], 'o-', color='green')
    ax1.set_xlabel('Number of PCA Components')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs PCA Components')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df_pca['Components'], df_pca['Time (s)'], 'o-', color='red')
    ax2.set_xlabel('Number of PCA Components')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time vs PCA Components')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp3_pca_comparison.svg', format='svg', bbox_inches='tight')
    plt.close()
    logger.log("PCA comparison plots saved")

# ===== EXPERIMENT 4: HYPERPARAMETER TUNING =====
if RUN_EXPERIMENTS['hyperparameter_tuning']:
    logger.log_separator()
    logger.log("EXPERIMENT 4: HYPERPARAMETER TUNING")
    logger.log_separator()
    
    # Use 90% PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    if SUBSET_SIZE:
        indices = np.random.choice(X_train_pca.shape[0], min(SUBSET_SIZE, X_train_pca.shape[0]), replace=False)
        X_sub = X_train_pca[indices]
        y_sub = y_train[indices]
    else:
        X_sub = X_train_pca
        y_sub = y_train
    
    # Linear SVM - C values
    logger.log("\n4a. Linear SVM - Testing different C values...")
    linear_results = []
    for C in EXPERIMENTS['linear_C']:
        logger.log(f"   Testing C={C}...")
        start = time.time()
        svm = LinearSVC(C=C, max_iter=1000, random_state=42)
        svm.fit(X_sub, y_sub)
        train_time = time.time() - start
        y_pred = svm.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        logger.log(f"      Accuracy: {acc:.4f}, Time: {train_time:.2f}s")
        linear_results.append(['Linear SVM', C, '-', train_time, acc])
    
    # RBF SVM - C and gamma values
    logger.log("\n4b. RBF SVM - Testing different C and gamma values...")
    rbf_results = []
    for C in EXPERIMENTS['rbf_C']:
        for gamma in EXPERIMENTS['rbf_gamma']:
            logger.log(f"   Testing C={C}, gamma={gamma}...")
            start = time.time()
            svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            svm.fit(X_sub, y_sub)
            train_time = time.time() - start
            y_pred = svm.predict(X_test_pca)
            acc = accuracy_score(y_test, y_pred)
            logger.log(f"      Accuracy: {acc:.4f}, Time: {train_time:.2f}s")
            rbf_results.append(['RBF SVM', C, gamma, train_time, acc])
    
    df_hyperparam = pd.DataFrame(linear_results + rbf_results,
                                columns=['Kernel', 'C', 'Gamma', 'Time (s)', 'Accuracy'])
    logger.save_results_table(df_hyperparam, 'hyperparameter_tuning')
    
    # Plot heatmap for RBF
    rbf_df = pd.DataFrame(rbf_results, columns=['Kernel', 'C', 'Gamma', 'Time (s)', 'Accuracy'])
    pivot_acc = rbf_df.pivot_table(values='Accuracy', index='C', columns='Gamma')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='YlOrRd')
    plt.title('RBF SVM: Accuracy for Different C and Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp4_hyperparameter_heatmap.svg', format='svg', bbox_inches='tight')
    plt.close()
    logger.log("Hyperparameter tuning heatmap saved")

# ===== EXPERIMENT 5: KNN COMPARISON =====
if RUN_EXPERIMENTS['knn_comparison']:
    logger.log_separator()
    logger.log("EXPERIMENT 5: K-NEAREST NEIGHBORS COMPARISON")
    logger.log_separator()
    
    # Use 90% PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    if SUBSET_SIZE:
        indices = np.random.choice(X_train_pca.shape[0], min(SUBSET_SIZE, X_train_pca.shape[0]), replace=False)
        X_sub = X_train_pca[indices]
        y_sub = y_train[indices]
    else:
        X_sub = X_train_pca
        y_sub = y_train
    
    knn_results = []
    
    for k in EXPERIMENTS['knn_neighbors']:
        logger.log(f"\nTesting k={k}...")
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_sub, y_sub)
        train_time = time.time() - start
        
        y_pred = knn.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        
        logger.log(f"   Accuracy: {acc:.4f}, Time: {train_time:.2f}s")
        knn_results.append([k, train_time, acc])
    
    df_knn = pd.DataFrame(knn_results, columns=['k', 'Time (s)', 'Accuracy'])
    logger.save_results_table(df_knn, 'knn_comparison')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_knn['k'], df_knn['Accuracy'], 'o-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('KNN Performance vs k Value', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp5_knn_comparison.svg', format='svg', bbox_inches='tight')
    plt.close()
    logger.log("KNN comparison plot saved")

# ===== EXPERIMENT 6: BINARY CLASSIFICATION =====
if RUN_EXPERIMENTS['binary_classification']:
    logger.log_separator()
    logger.log("EXPERIMENT 6: BINARY CLASSIFICATION TASKS")
    logger.log_separator()
    
    # Test on different class pairs
    class_pairs = [(0, 1), (2, 3), (8, 9)]  # airplane vs car, bird vs cat, ship vs truck
    
    # Use 90% PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    binary_results = []
    
    for class_a, class_b in class_pairs:
        logger.log(f"\nBinary task: {class_names[class_a]} vs {class_names[class_b]}")
        
        # Filter training data
        train_mask = (y_train == class_a) | (y_train == class_b)
        X_train_binary = X_train_pca[train_mask]
        y_train_binary = y_train[train_mask]
        
        # Filter test data
        test_mask = (y_test == class_a) | (y_test == class_b)
        X_test_binary = X_test_pca[test_mask]
        y_test_binary = y_test[test_mask]
        
        logger.log(f"   Training samples: {len(y_train_binary)}, Test samples: {len(y_test_binary)}")
        
        # Train Linear SVM
        start = time.time()
        svm_linear = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        svm_linear.fit(X_train_binary, y_train_binary)
        train_time = time.time() - start
        y_pred = svm_linear.predict(X_test_binary)
        acc_linear = accuracy_score(y_test_binary, y_pred)
        
        # Train RBF SVM
        start = time.time()
        svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        svm_rbf.fit(X_train_binary, y_train_binary)
        train_time_rbf = time.time() - start
        y_pred_rbf = svm_rbf.predict(X_test_binary)
        acc_rbf = accuracy_score(y_test_binary, y_pred_rbf)
        
        logger.log(f"   Linear SVM - Acc: {acc_linear:.4f}, Time: {train_time:.2f}s")
        logger.log(f"   RBF SVM    - Acc: {acc_rbf:.4f}, Time: {train_time_rbf:.2f}s")
        
        binary_results.append([f"{class_names[class_a]} vs {class_names[class_b]}", 
                              acc_linear, acc_rbf])
    
    df_binary = pd.DataFrame(binary_results, 
                            columns=['Task', 'Linear SVM Acc', 'RBF SVM Acc'])
    logger.save_results_table(df_binary, 'binary_classification')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_binary))
    width = 0.35
    ax.bar(x - width/2, df_binary['Linear SVM Acc'], width, label='Linear SVM', alpha=0.8)
    ax.bar(x + width/2, df_binary['RBF SVM Acc'], width, label='RBF SVM', alpha=0.8)
    ax.set_xlabel('Binary Classification Task')
    ax.set_ylabel('Accuracy')
    ax.set_title('Binary Classification Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(df_binary['Task'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp6_binary_classification.svg', format='svg', bbox_inches='tight')
    plt.close()