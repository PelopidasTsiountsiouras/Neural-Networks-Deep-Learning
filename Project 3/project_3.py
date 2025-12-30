import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import time
import csv
import os
from datetime import datetime
import logging

# Create directories for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Setup logging
log_filename = f'logs/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')


# ==================== AUTOENCODER ARCHITECTURES ====================

class Autoencoder(nn.Module):
    """Simple Autoencoder with configurable architecture"""
    def __init__(self, input_size=784, encoding_dim=32, hidden_dims=None):
        super(Autoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_size))
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class DigitClassifier(nn.Module):
    """Simple classifier to test if reconstructed digits are recognizable"""
    def __init__(self, input_size=784):
        super(DigitClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)


# ==================== DATA LOADING ====================

def load_mnist_data(train_split=0.6):
    """Load MNIST and split into train/test"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training data into train (60%) and validation (40%)
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logging.info(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}')
    
    return train_dataset, val_dataset, test_dataset


# ==================== ACCURACY CALCULATION ====================

def calculate_reconstruction_accuracy(original, reconstructed, threshold=0.1):
    """
    Calculate reconstruction accuracy as percentage of pixels 
    reconstructed within threshold of original
    """
    pixel_diff = np.abs(original - reconstructed)
    correct_pixels = (pixel_diff < threshold).sum()
    total_pixels = original.size
    accuracy = 100 * correct_pixels / total_pixels
    return accuracy


def calculate_image_accuracy(original, reconstructed, threshold=0.1):
    """
    Calculate percentage of images that are well-reconstructed
    (all pixels within threshold)
    """
    pixel_diff = np.abs(original - reconstructed)
    # Check if all pixels in each image are within threshold
    max_diff_per_image = np.max(pixel_diff, axis=1)
    well_reconstructed = (max_diff_per_image < threshold).sum()
    accuracy = 100 * well_reconstructed / len(original)
    return accuracy


# ==================== TRAINING FUNCTIONS ====================

def train_autoencoder(model, train_loader, val_loader, criterion, optimizer, epochs, model_name):
    """Train autoencoder and return training history"""
    logging.info(f'\n{"="*60}')
    logging.info(f'Training {model_name}')
    logging.info(f'{"="*60}')
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy for this batch
            batch_acc = calculate_reconstruction_accuracy(
                data.cpu().numpy(), 
                output.detach().cpu().numpy(),
                threshold=0.1
            )
            train_acc += batch_acc
            train_batches += 1
        
        train_loss /= len(train_loader)
        train_acc /= train_batches
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.view(data.size(0), -1).to(device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
                
                # Calculate accuracy for this batch
                batch_acc = calculate_reconstruction_accuracy(
                    data.cpu().numpy(), 
                    output.cpu().numpy(),
                    threshold=0.1
                )
                val_acc += batch_acc
                val_batches += 1
        
        val_loss /= len(val_loader)
        val_acc /= val_batches
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%')
    
    training_time = time.time() - start_time
    logging.info(f'Training completed in {training_time:.2f} seconds')
    logging.info(f'Final Train Accuracy: {train_accuracies[-1]:.2f}%')
    logging.info(f'Final Val Accuracy: {val_accuracies[-1]:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies, training_time


def train_classifier(classifier, train_loader, val_loader, epochs=20):
    """Train the digit classifier"""
    logging.info('\nTraining digit classifier...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    best_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        for data, labels in train_loader:
            data = data.view(data.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.view(data.size(0), -1).to(device)
                labels = labels.to(device)
                output = classifier(data)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), 'models/digit_classifier.pth')
        
        if (epoch + 1) % 5 == 0:
            logging.info(f'Classifier Epoch [{epoch+1}/{epochs}], Accuracy: {acc:.2f}%')
    
    logging.info(f'Best classifier accuracy: {best_acc:.2f}%')
    return best_acc


# ==================== EVALUATION FUNCTIONS ====================

def evaluate_reconstruction(model, data_loader, criterion):
    """Evaluate reconstruction quality with multiple metrics"""
    model.eval()
    total_loss = 0.0
    total_pixel_acc = 0.0
    total_image_acc = 0.0
    all_originals = []
    all_reconstructed = []
    n_batches = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()
            
            data_np = data.cpu().numpy()
            output_np = output.cpu().numpy()
            
            # Calculate accuracies
            pixel_acc = calculate_reconstruction_accuracy(data_np, output_np, threshold=0.1)
            image_acc = calculate_image_accuracy(data_np, output_np, threshold=0.1)
            
            total_pixel_acc += pixel_acc
            total_image_acc += image_acc
            n_batches += 1
            
            all_originals.append(data_np)
            all_reconstructed.append(output_np)
    
    avg_loss = total_loss / len(data_loader)
    avg_pixel_acc = total_pixel_acc / n_batches
    avg_image_acc = total_image_acc / n_batches
    
    all_originals = np.vstack(all_originals)
    all_reconstructed = np.vstack(all_reconstructed)
    
    mse = mean_squared_error(all_originals, all_reconstructed)
    
    return avg_loss, mse, avg_pixel_acc, avg_image_acc, all_originals, all_reconstructed


def evaluate_with_classifier(classifier, reconstructed_data, original_labels):
    """Test if classifier can recognize reconstructed digits"""
    classifier.eval()
    reconstructed_tensor = torch.FloatTensor(reconstructed_data).to(device)
    
    with torch.no_grad():
        outputs = classifier(reconstructed_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    predicted = predicted.cpu().numpy()
    correct = (predicted == original_labels).sum()
    accuracy = 100 * correct / len(original_labels)
    
    return accuracy, predicted


def find_best_worst_reconstructions(originals, reconstructed, n_examples=10):
    """Find best and worst reconstruction examples"""
    mse_per_sample = np.mean((originals - reconstructed) ** 2, axis=1)
    
    best_indices = np.argsort(mse_per_sample)[:n_examples]
    worst_indices = np.argsort(mse_per_sample)[-n_examples:]
    
    return best_indices, worst_indices, mse_per_sample


# ==================== PCA COMPARISON ====================

def train_pca_reconstruction(train_data, test_data, n_components_list):
    """Train PCA and reconstruct data"""
    logging.info('\n' + '='*60)
    logging.info('PCA Reconstruction')
    logging.info('='*60)
    
    results = {}
    
    for n_components in n_components_list:
        logging.info(f'\nPCA with {n_components} components')
        start_time = time.time()
        
        pca = PCA(n_components=n_components)
        pca.fit(train_data)
        
        # Reconstruct
        train_transformed = pca.transform(train_data)
        train_reconstructed = pca.inverse_transform(train_transformed)
        
        test_transformed = pca.transform(test_data)
        test_reconstructed = pca.inverse_transform(test_transformed)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        train_mse = mean_squared_error(train_data, train_reconstructed)
        test_mse = mean_squared_error(test_data, test_reconstructed)
        
        # Calculate accuracies
        train_pixel_acc = calculate_reconstruction_accuracy(train_data, train_reconstructed, threshold=0.1)
        test_pixel_acc = calculate_reconstruction_accuracy(test_data, test_reconstructed, threshold=0.1)
        
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        results[n_components] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_pixel_accuracy': train_pixel_acc,
            'test_pixel_accuracy': test_pixel_acc,
            'variance_explained': variance_explained,
            'training_time': training_time,
            'reconstructed': test_reconstructed
        }
        
        logging.info(f'Training time: {training_time:.2f}s')
        logging.info(f'Train MSE: {train_mse:.6f}, Train Pixel Acc: {train_pixel_acc:.2f}%')
        logging.info(f'Test MSE: {test_mse:.6f}, Test Pixel Acc: {test_pixel_acc:.2f}%')
        logging.info(f'Variance explained: {variance_explained:.4f}')
    
    return results


# ==================== VISUALIZATION ====================

def plot_training_history(train_losses, val_losses, train_accs, val_accs, model_name, save_path):
    """Plot training and validation loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss - {model_name}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'Reconstruction Accuracy - {model_name}', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved training history plot: {save_path}')


def plot_reconstructions(originals, reconstructed, indices, title, save_path, n_show=10):
    """Plot original vs reconstructed images"""
    n_show = min(n_show, len(indices))
    fig, axes = plt.subplots(2, n_show, figsize=(20, 4))
    
    for i, idx in enumerate(indices[:n_show]):
        # Original
        axes[0, i].imshow(originals[idx].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[idx].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved reconstruction plot: {save_path}')


def plot_comparison_all_models(results_dict, metric_name, save_path):
    """Plot comparison of all models"""
    plt.figure(figsize=(12, 6))
    
    models = list(results_dict.keys())
    values = [results_dict[model][metric_name] for model in models]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    bars = plt.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Comparison: {metric_name.replace("_", " ").title()}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved comparison plot: {save_path}')


# ==================== SAVE RESULTS ====================

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    filepath = f'results/{filename}'
    
    if not results:
        logging.warning(f'No results to save for {filename}')
        return
    
    with open(filepath, 'w', newline='') as f:
        # Get all unique keys from all result dictionaries
        all_keys = set()
        for result in results.values():
            all_keys.update(result.keys())
        
        writer = csv.DictWriter(f, fieldnames=['model'] + sorted(all_keys))
        writer.writeheader()
        
        for model_name, result in results.items():
            row = {'model': model_name}
            row.update(result)
            writer.writerow(row)
    
    logging.info(f'Saved results to: {filepath}')


def save_epoch_history_to_csv(model_name, train_losses, val_losses, train_accs, val_accs):
    """Save epoch-by-epoch training history to CSV"""
    filepath = f'results/training_history_{model_name}.csv'
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
        
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                train_losses[epoch],
                val_losses[epoch],
                train_accs[epoch],
                val_accs[epoch]
            ])
    
    logging.info(f'Saved epoch history to: {filepath}')


# ==================== MAIN EXPERIMENT ====================

def main():
    logging.info('='*60)
    logging.info('MNIST AUTOENCODER PROJECT')
    logging.info('='*60)
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_mnist_data(train_split=0.6)
    
    # Hyperparameters to test
    batch_sizes = [64, 128]
    learning_rates = [0.001, 0.0001]
    epochs = 50
    
    # Architecture configurations
    architectures = {
        'AE_Small': {'encoding_dim': 32, 'hidden_dims': [256, 128]},
        'AE_Medium': {'encoding_dim': 64, 'hidden_dims': [512, 256]},
        'AE_Large': {'encoding_dim': 128, 'hidden_dims': [512, 256, 128]}
    }
    
    all_results = {}
    
    # Train autoencoders with different configurations
    for arch_name, arch_config in architectures.items():
        for batch_size in batch_sizes:
            for lr in learning_rates:
                model_name = f'{arch_name}_bs{batch_size}_lr{lr}'
                
                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Create model
                model = Autoencoder(**arch_config).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Train
                train_losses, val_losses, train_accs, val_accs, training_time = train_autoencoder(
                    model, train_loader, val_loader, criterion, optimizer, epochs, model_name
                )
                
                # Save epoch history to CSV
                save_epoch_history_to_csv(model_name, train_losses, val_losses, train_accs, val_accs)
                
                # Plot training history
                plot_training_history(
                    train_losses, val_losses, train_accs, val_accs, model_name,
                    f'plots/training_history_{model_name}.png'
                )
                
                # Evaluate on test set
                test_loss, test_mse, test_pixel_acc, test_image_acc, test_originals, test_reconstructed = evaluate_reconstruction(
                    model, test_loader, criterion
                )
                
                logging.info(f'\n{model_name} Test Results:')
                logging.info(f'Test Loss: {test_loss:.6f}')
                logging.info(f'Test MSE: {test_mse:.6f}')
                logging.info(f'Test Pixel Accuracy: {test_pixel_acc:.2f}%')
                logging.info(f'Test Image Accuracy: {test_image_acc:.2f}%')
                
                # Find best and worst reconstructions
                best_idx, worst_idx, mse_per_sample = find_best_worst_reconstructions(
                    test_originals, test_reconstructed, n_examples=10
                )
                
                # Plot examples
                plot_reconstructions(
                    test_originals, test_reconstructed, best_idx,
                    f'Best Reconstructions - {model_name}',
                    f'plots/best_reconstructions_{model_name}.png'
                )
                
                plot_reconstructions(
                    test_originals, test_reconstructed, worst_idx,
                    f'Worst Reconstructions - {model_name}',
                    f'plots/worst_reconstructions_{model_name}.png'
                )
                
                # Save model
                torch.save(model.state_dict(), f'models/{model_name}.pth')
                
                # Store results
                all_results[model_name] = {
                    'architecture': arch_name,
                    'encoding_dim': arch_config['encoding_dim'],
                    'hidden_dims': str(arch_config['hidden_dims']),
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'epochs': epochs,
                    'training_time_seconds': training_time,
                    'final_train_loss': train_losses[-1],
                    'final_val_loss': val_losses[-1],
                    'final_train_accuracy': train_accs[-1],
                    'final_val_accuracy': val_accs[-1],
                    'test_loss': test_loss,
                    'test_mse': test_mse,
                    'test_pixel_accuracy': test_pixel_acc,
                    'test_image_accuracy': test_image_acc,
                    'best_reconstruction_mse': mse_per_sample[best_idx].mean(),
                    'worst_reconstruction_mse': mse_per_sample[worst_idx].mean()
                }
    
    # Train classifier on original digits
    logging.info('\n' + '='*60)
    logging.info('Training Digit Classifier')
    logging.info('='*60)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    classifier = DigitClassifier().to(device)
    classifier_acc = train_classifier(classifier, train_loader, val_loader, epochs=20)
    
    # Test classifier on reconstructed digits from all models
    logging.info('\n' + '='*60)
    logging.info('Testing Classifier on Reconstructed Digits')
    logging.info('='*60)
    
    # Get original labels
    test_labels = []
    test_data_flat = []
    for data, labels in test_loader:
        test_labels.extend(labels.numpy())
        test_data_flat.append(data.view(data.size(0), -1).numpy())
    test_labels = np.array(test_labels)
    test_data_flat = np.vstack(test_data_flat)
    
    for model_name in all_results.keys():
        # Load model
        arch_name = all_results[model_name]['architecture']
        arch_config = architectures[arch_name]
        model = Autoencoder(**arch_config).to(device)
        model.load_state_dict(torch.load(f'models/{model_name}.pth'))
        model.eval()
        
        # Reconstruct test data
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_data_flat).to(device)
            reconstructed = model(test_tensor).cpu().numpy()
        
        # Test classifier
        recon_acc, predictions = evaluate_with_classifier(classifier, reconstructed, test_labels)
        all_results[model_name]['classifier_accuracy_on_reconstructed'] = recon_acc
        
        logging.info(f'{model_name}: Classifier accuracy on reconstructed = {recon_acc:.2f}%')
    
    # PCA Comparison
    logging.info('\n' + '='*60)
    logging.info('PCA Comparison')
    logging.info('='*60)
    
    # Prepare data for PCA
    train_data_flat = []
    for data, _ in train_loader:
        train_data_flat.append(data.view(data.size(0), -1).numpy())
    train_data_flat = np.vstack(train_data_flat)
    
    # Test different PCA components
    pca_components = [32, 64, 128, 256]
    pca_results = train_pca_reconstruction(train_data_flat, test_data_flat, pca_components)
    
    # Test classifier on PCA reconstructions
    for n_comp, pca_result in pca_results.items():
        recon_acc, _ = evaluate_with_classifier(
            classifier, pca_result['reconstructed'], test_labels
        )
        pca_result['classifier_accuracy_on_reconstructed'] = recon_acc
        logging.info(f'PCA-{n_comp}: Classifier accuracy on reconstructed = {recon_acc:.2f}%')
    
    # Add PCA results to all_results
    for n_comp, pca_result in pca_results.items():
        model_name = f'PCA_{n_comp}'
        all_results[model_name] = {
            'architecture': 'PCA',
            'encoding_dim': n_comp,
            'hidden_dims': 'N/A',
            'batch_size': 'N/A',
            'learning_rate': 'N/A',
            'epochs': 'N/A',
            'training_time_seconds': pca_result['training_time'],
            'final_train_loss': 'N/A',
            'final_val_loss': 'N/A',
            'final_train_accuracy': pca_result['train_pixel_accuracy'],
            'final_val_accuracy': 'N/A',
            'test_loss': 'N/A',
            'test_mse': pca_result['test_mse'],
            'test_pixel_accuracy': pca_result['test_pixel_accuracy'],
            'test_image_accuracy': 'N/A',
            'best_reconstruction_mse': 'N/A',
            'worst_reconstruction_mse': 'N/A',
            'classifier_accuracy_on_reconstructed': pca_result['classifier_accuracy_on_reconstructed'],
            'variance_explained': pca_result['variance_explained']
        }
    
    # Save all results to main CSV
    save_results_to_csv(all_results, 'all_results.csv')
    
    # Create separate CSV for just training/testing accuracies
    accuracy_results = {}
    for model_name, results in all_results.items():
        accuracy_results[model_name] = {
            'train_accuracy': results.get('final_train_accuracy', 'N/A'),
            'test_pixel_accuracy': results.get('test_pixel_accuracy', 'N/A'),
            'test_image_accuracy': results.get('test_image_accuracy', 'N/A'),
            'classifier_accuracy_on_reconstructed': results.get('classifier_accuracy_on_reconstructed', 'N/A')
        }
    save_results_to_csv(accuracy_results, 'accuracy_summary.csv')
    
    # Create comparison plots
    ae_results = {k: v for k, v in all_results.items() if 'AE_' in k}
    pca_only_results = {k: v for k, v in all_results.items() if 'PCA_' in k}
    
    # Plot MSE comparison
    comparison_data = {}
    for model_name, results in all_results.items():
        if results['test_mse'] != 'N/A':
            comparison_data[model_name] = {'test_mse': results['test_mse']}
    
    if comparison_data:
        plot_comparison_all_models(comparison_data, 'test_mse', 'plots/comparison_mse.png')
    
    # Plot pixel accuracy comparison
    pixel_acc_data = {}
    for model_name, results in all_results.items():
        if results.get('test_pixel_accuracy') != 'N/A':
            pixel_acc_data[model_name] = {'test_pixel_accuracy': results['test_pixel_accuracy']}
    
    if pixel_acc_data:
        plot_comparison_all_models(pixel_acc_data, 'test_pixel_accuracy', 'plots/comparison_pixel_accuracy.png')
    
    # Plot training time comparison
    time_data = {}
    for model_name, results in all_results.items():
        if results['training_time_seconds'] != 'N/A':
            time_data[model_name] = {'training_time_seconds': results['training_time_seconds']}
    
    if time_data:
        plot_comparison_all_models(time_data, 'training_time_seconds', 'plots/comparison_training_time.png')
    
    # Plot classifier accuracy comparison
    classifier_data = {}
    for model_name, results in all_results.items():
        if 'classifier_accuracy_on_reconstructed' in results:
            classifier_data[model_name] = {'classifier_accuracy': results['classifier_accuracy_on_reconstructed']}
    
    if classifier_data:
        plot_comparison_all_models(
            classifier_data, 'classifier_accuracy',
            'plots/comparison_classifier_accuracy.png'
        )
    
    # Final summary
    logging.info('\n' + '='*60)
    logging.info('EXPERIMENT COMPLETED')
    logging.info('='*60)
    logging.info(f'\nResults saved to:')
    logging.info(f'  - results/all_results.csv (complete results)')
    logging.info(f'  - results/accuracy_summary.csv (accuracy summary)')
    logging.info(f'  - results/training_history_*.csv (epoch-by-epoch histories)')
    logging.info(f'Models saved to: models/')
    logging.info(f'Plots saved to: plots/')
    logging.info(f'Log file: {log_filename}')
    
    # Find best model based on test pixel accuracy
    ae_models = {k: v for k, v in all_results.items() if 'AE_' in k}
    best_model = max(ae_models.items(), key=lambda x: x[1]['test_pixel_accuracy'])
    logging.info(f'\nBest Autoencoder Model (by pixel accuracy): {best_model[0]}')
    logging.info(f'Test Pixel Accuracy: {best_model[1]["test_pixel_accuracy"]:.2f}%')
    logging.info(f'Test MSE: {best_model[1]["test_mse"]:.6f}')
    logging.info(f'Classifier Accuracy on Reconstructed: {best_model[1]["classifier_accuracy_on_reconstructed"]:.2f}%')
    
    print('\n' + '='*60)
    print('All experiments completed successfully!')
    print('='*60)


if __name__ == '__main__':
    main()