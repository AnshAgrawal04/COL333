import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import norm, multivariate_normal
import pickle
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

def set_seed(SEED):

    # Set NumPy seed
    np.random.seed(SEED)

    # Set PyTorch seed for CPU
    torch.manual_seed(SEED)

    # Set PyTorch seed for CUDA (if using a GPU)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU setups
    torch.manual_seed(SEED)

    # Ensure PyTorch operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.epochs = 50
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # out: 32x14x14
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1) # out: 64x7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 256)
        self.fc3 = nn.Linear(256, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # upsample to 32x14x14
        self.deconv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1)  # upsample to 1x28x28

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        h = F.relu(self.fc1(x.view(-1, 64 * 7 * 7)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        h = F.relu(self.fc3(h)).view(-1, 64, 7, 7)
        h = F.relu(self.deconv1(h))
        return torch.sigmoid(self.deconv2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GMM:
    def __init__(self, clusters=None, epochs=50, eps=1e-10):
        self.mu = {}
        self.sigma = {}
        self.phi = {}
        self.clusters = clusters if clusters is not None else [1, 4, 8]
        self.epochs = epochs
        self.eps = eps
    
    def density(self, x, y):
        if y not in self.phi or y not in self.mu or y not in self.sigma:
            raise ValueError(f"Cluster {y} not initialized in phi, mu, or sigma")
        
        t1 = self.phi[y]
        cov_matrix = self.sigma[y] + self.eps * np.eye(self.sigma[y].shape[0])
        t2 = multivariate_normal.pdf(x.flatten(), mean=self.mu[y].flatten(), cov=cov_matrix)
        return t1 * t2

    def train(self, X):
        m = len(X)
        r = np.zeros((m, len(self.clusters)))
        accs = []
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            
            # Expectation Step
            for i, x in enumerate(X):
                for c, cluster in enumerate(self.clusters):
                    r[i, c] = self.density(x, cluster)
                r[i] /= (np.sum(r[i]) + self.eps)
            
            # Maximization Step
            for c, cluster in enumerate(self.clusters):
                responsibility_sum = np.sum(r[:, c])
                if responsibility_sum == 0:
                    continue
                
                self.phi[cluster] = responsibility_sum / m
                self.mu[cluster] = sum(r[i, c] * X[i] for i in range(len(X))) / responsibility_sum
                self.sigma[cluster] = sum(r[i, c] * np.outer(X[i] - self.mu[cluster], X[i] - self.mu[cluster]) 
                                          for i in range(len(X))) / responsibility_sum
            
            # preds = self.predict(X)
            # accuracy = (preds == y).sum() / len(y)
            # accs.append(accuracy)
            # print(f'Accuracy: {100 * accuracy: .2f}')
        # Separate figure for accuracy vs epochs plot
        # plt.figure(figsize=(8, 6))  # New figure for accuracy plot
        # plt.plot(range(1, self.epochs + 1), accs, marker='o', label='Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy vs Epochs')
        # plt.legend()
        # plt.savefig('benchmark/cnn1_100eps.png')

    def predict(self, X):
        preds = []
        for x in X:
            predictions = [self.density(x, cluster) for cluster in self.clusters]
            pred_idx = np.argmax(predictions)
            preds.append(self.clusters[pred_idx])
        return preds
    
    def initialise(self, Xv, yv):
        labels = np.unique(yv)
        Xv = np.array(Xv)
        
        for label in labels:
            label_data = Xv[yv == label]
            self.mu[label] = np.mean(label_data, axis=0)
            self.phi[label] = len(label_data) / len(Xv)
            self.sigma[label] = np.mean([(x - self.mu[label]) @ (x - self.mu[label]).T for x in label_data], axis=0) + self.eps * np.eye(Xv.shape[1])
        self.clusters = list(labels)

    def save_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'mu': self.mu, 'sigma': self.sigma, 'phi': self.phi}, f)
    
    def load_params(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.mu = params['mu']
            self.sigma = params['sigma']
            self.phi = params['phi']

class NPZDataset(Dataset):
    def __init__(self, npz_path, transform=None, keep=[1, 4, 8]):
        data = np.load(npz_path)
        self.images = data['data']
        self.labels = data['labels'] if 'labels' in data else np.ones(len(self.images))
        self.indices = [i for i in range(len(self.labels)) if self.labels[i] in keep]
        self.images = self.images[self.indices]
        self.labels = self.labels[self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def loss_function(recon_x, x, mu, logvar):
    BETA = 0.5
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -BETA * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device='cpu'):
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    vae.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                generated_img = vae.decode(z_sample).cpu().view(digit_size, digit_size).numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = generated_img
    # print(figure)
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')
    plt.axis('off')
    plt.savefig('manifold.png')

def show_reconstruction(vaePath, val_loader, n=15):
    model = VAE().to(device)
    model.load_state_dict(torch.load(vaePath))
    recon_images_np = []
    gt_data = []
    model.eval()

    for data, _ in val_loader:
        data = data.to(device)
        gt_data.append(data.cpu().numpy())
        with torch.no_grad():
            recon_data, _, _ = model(data)
            recon_images_np.append(recon_data.cpu().view(-1, 28, 28).numpy())

    recon_images_np = np.concatenate(recon_images_np, axis=0)
    gt_data = np.concatenate(gt_data, axis=0)
    np.savez('vae_reconstructed.npz', data=recon_images_np)
    
    fig, axes = plt.subplots(2, n, figsize=(30, 4))
    for i in range(n):
        # Original images
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(recon_data[i].cpu().view(28, 28).detach().numpy(), cmap='gray')
        axes[1, i].axis('off')
    
    plt.savefig('reconstruction.png')
    plot_2d_manifold(model)

def evaluate_reconstruction(predicted_images, ground_truth_images):
    if isinstance(predicted_images, torch.Tensor):
        predicted_images = predicted_images.cpu().numpy()
    if isinstance(ground_truth_images, torch.Tensor):
        ground_truth_images = ground_truth_images.cpu().numpy()

    mse_scores = []
    ssim_scores = []

    for pred_img, gt_img in zip(predicted_images, ground_truth_images):

        mse = np.mean((pred_img - gt_img) ** 2)
        mse_scores.append(1 - mse)

        ssim_val = ssim(pred_img, gt_img, data_range=1)
        ssim_scores.append(ssim_val)

    mse_score = np.mean(mse_scores)
    ssim_score = np.mean(ssim_scores)

    combined_score = (mse_score + ssim_score) / 2

    return combined_score, mse_score, ssim_score

def evaluate_recon_score(recon_file, gt_file):
    # Load reconstructed and ground truth images
    recon_data = np.load(recon_file)['data']
    gt_data = np.load(gt_file)['data']/255.0

    # Evaluate reconstruction scores
    combined_score, mse_score, ssim_score = evaluate_reconstruction(recon_data, gt_data)

    # Print scores
    print(f'Combined Score: {combined_score:.4f}, (1 - MSE) Score: {mse_score:.4f}, SSIM Score: {ssim_score:.4f}')
    
def evaluate_gmm_performance(labels_true, labels_pred):
    
    accuracy = accuracy_score(labels_true, labels_pred)
    precision_macro = precision_score(labels_true, labels_pred, average='macro')  # Macro precision
    recall_macro = recall_score(labels_true, labels_pred, average='macro')  # Macro recall
    f1_macro = f1_score(labels_true, labels_pred, average='macro')  # Macro F1

    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

def print_accuracy(gmm, X, y):
    Xt = [np.array(x).reshape(2, 1) for x in X]
    preds = gmm.predict(Xt)
    print(f'Accuracy: {(preds == y).sum() / len(y)}')

def plot_clusters(X, y, gmm):
    clusters = gmm.clusters
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    # Xgmm = np.array([np.array(x).reshape(2, 1) for x in X])
    # y = np.array(gmm.predict(Xgmm))
    # ellipse_colors = ['yellow', 'cyan', 'magenta']
    for i, c in enumerate(clusters):
        cluster_points = np.where(y == c)[0]
        plt.scatter(X[cluster_points][:, 0], X[cluster_points][:, 1], label = f'Cluster {c}', color=colors[i], s=1,
                    alpha=0.4)
    for i, c in enumerate(clusters):
        mu = gmm.mu[c].flatten()
        sigma = gmm.sigma[c]
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = patches.Ellipse(mu, width, height, angle=angle, edgecolor=colors[i], facecolor='none', 
                      label=f'Cluster {c} Ellipse', linewidth=3)
        plt.gca().add_patch(ellipse)
        plt.scatter(mu[0], mu[1], color='black', marker='x', s=100, label=f'Cluster {c} Mean')
    plt.legend()
    plt.savefig('gmm_og_clusters.png')

def train(train_loader, val_loader, vaePath, gmmPath):
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    for epoch in range(vae.epochs):
        vae.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        
        scheduler.step()

        vae.eval()
        val_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = vae(data)
                val_loss += loss_function(recon_batch, data, mu, logvar).item()

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(vae.state_dict(), vaePath)
    # vae.load_state_dict(torch.load(vaePath))

    vae.eval()
    Xt = []
    Xv = []
    # yt = []
    yv = []

    with torch.no_grad():
        for data, label in train_loader:
            data = data.to(device)
            mu, _ = vae.encode(data)
            Xt.append(mu.cpu().numpy())
            # yt.append(label)

        for data, label in val_loader:
            data = data.to(device)
            mu, _ = vae.encode(data)
            Xv.append(mu.cpu().numpy())
            yv.append(label)

    Xt = np.concatenate(Xt, axis=0)
    Xv = np.concatenate(Xv, axis=0)
    yv = np.concatenate(yv, axis=0)
    # yt = np.concatenate(yt, axis=0)
    train_vectors = [np.array(x).reshape(2, 1) for x in Xt]
    val_vectors = [np.array(x).reshape(2, 1) for x in Xv]

    gmm = GMM()
    gmm.initialise(val_vectors, yv)
    gmm.train(train_vectors)
    gmm.save_params(gmmPath)
    # gmm.load_params(gmmPath)
    # plot_clusters(Xt, yt, gmm)

def classify(vaePath, gmmPath, test_loader):
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(vaePath))
    test_vectors = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            mu, _ = vae.encode(data)
            test_vectors.append(mu.cpu().numpy())
    test_vectors = np.concatenate(test_vectors, axis=0)
    test_vectors = [np.array(x).reshape(2, 1) for x in test_vectors]
    gmm = GMM()
    gmm.load_params(gmmPath)
    preds = gmm.predict(test_vectors)
    pred_df = pd.DataFrame(preds, columns=['Predicted_label'])
    pred_df.to_csv('vae.csv', index=False)

def give_accuracy(pred_file, data_loader):
    pred = pd.read_csv(pred_file)['Predicted_label'].values
    y = []
    with torch.no_grad():
        for _, label in data_loader:
            y.append(label)
    labels = np.concatenate(y, axis=0)
    print(f'Accuracy: {np.mean(labels == pred)}')

if __name__ == "__main__":
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None

    if len(sys.argv) == 4:  # Reconstruction
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3
        val_dataset = NPZDataset(path_to_test_dataset_recon, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        show_reconstruction(vaePath, val_loader, n=len(val_dataset))
        # evaluate_recon_score('vae_reconstructed.npz', path_to_test_dataset_recon)

    elif len(sys.argv) == 5:  # Classification
        path_to_test_dataset = arg1
        vaePath = arg3
        gmmPath = arg4
        test_dataset = NPZDataset(path_to_test_dataset, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        classify(vaePath, gmmPath, test_loader)
        # print(evaluate_gmm_performance(test_dataset.labels, pd.read_csv('vae.csv')['Predicted_label'].values))
        # give_accuracy('vae.csv', test_loader)
        
    elif len(sys.argv) == 6:  # Training
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5
        
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = NPZDataset(path_to_train_dataset, transform=transform)
        val_dataset = NPZDataset(path_to_val_dataset, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        train(train_loader, val_loader, vaePath, gmmPath)

    else:
        path_to_train_dataset = 'data/mnist_1_4_8_all.npz'
        path_to_val_dataset = 'data/mnist_1_4_8_val_recon.npz'
        trainStatus = arg3
        vaePath = 'vae.pth'
        gmmPath = 'gmm_params.pkl'

        # Prepare data loaders
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = NPZDataset(path_to_train_dataset, transform=transform)
        split_ratio = 0.8
        train_size = int(split_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, lengths=[train_size, test_size])
        val_dataset = NPZDataset(path_to_val_dataset, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        train(train_loader, val_loader, vaePath, gmmPath)
        show_reconstruction(vaePath, val_loader)
        classify(vaePath, gmmPath, test_loader)
        give_accuracy('vae.csv', test_loader)