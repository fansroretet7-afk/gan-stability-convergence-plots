import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math



# 1. Генерация данных: 2D Gaussian Mixture
def get_gmm_samples(batch_size, radius=2.0, num_modes=8, std=0.1):
    """Генерирует батч точек из смеси 8 гауссиан, расположенных по кругу."""
    modes = np.random.randint(0, num_modes, batch_size)
    angles = modes * (2 * np.pi / num_modes)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    noise = np.random.randn(batch_size, 2) * std
    samples = centers + noise
    return torch.tensor(samples, dtype=torch.float32)


# 2. Архитектуры Нейросетей
class Generator(nn.Module):
    def __init__(self, latent_dim=2, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)



# 3. Функция обучения с выбором метода
def train_toy_gan(method='GDA', iterations=5000, batch_size=256, lr=0.01):
    print(f"Запуск обучения методом: {method}")

    G = Generator()
    D = Discriminator()
    criterion = nn.BCEWithLogitsLoss()

    if method == 'OGD':
        for p in list(G.parameters()) + list(D.parameters()):
            p.prev_grad = torch.zeros_like(p.data)

    for step in range(iterations):
        real_data = get_gmm_samples(batch_size)
        z = torch.randn(batch_size, 2)

        # --- Vanilla GDA ---
        if method == 'GDA':
            fake_data = G(z)
            D_loss = criterion(D(real_data), torch.ones(batch_size, 1)) + \
                     criterion(D(fake_data.detach()), torch.zeros(batch_size, 1))
            G.zero_grad();
            D.zero_grad()
            D_loss.backward()
            for p in D.parameters(): p.data.add_(-lr, p.grad.data)

            G_loss = criterion(D(fake_data), torch.ones(batch_size, 1))
            G.zero_grad();
            D.zero_grad()
            G_loss.backward()
            for p in G.parameters(): p.data.add_(-lr, p.grad.data)

        # --- Extragradient (EG) ---
        elif method == 'EG':
            fake_data = G(z)
            D_loss = criterion(D(real_data), torch.ones(batch_size, 1)) + \
                     criterion(D(fake_data.detach()), torch.zeros(batch_size, 1))
            G_loss = criterion(D(fake_data), torch.ones(batch_size, 1))
            G.zero_grad();
            D.zero_grad()
            D_loss.backward();
            G_loss.backward()

            for p in list(G.parameters()) + list(D.parameters()):
                p.saved_data = p.data.clone()
                p.data.add_(-lr, p.grad.data)

            fake_data_ext = G(z)
            D_loss_ext = criterion(D(real_data), torch.ones(batch_size, 1)) + \
                         criterion(D(fake_data_ext.detach()), torch.zeros(batch_size, 1))
            G_loss_ext = criterion(D(fake_data_ext), torch.ones(batch_size, 1))
            G.zero_grad();
            D.zero_grad()
            D_loss_ext.backward();
            G_loss_ext.backward()

            for p in list(G.parameters()) + list(D.parameters()):
                p.data = p.saved_data
                p.data.add_(-lr, p.grad.data)

        # --- Optimistic GD (OGD) ---
        elif method == 'OGD':
            fake_data = G(z)
            D_loss = criterion(D(real_data), torch.ones(batch_size, 1)) + \
                     criterion(D(fake_data.detach()), torch.zeros(batch_size, 1))
            G_loss = criterion(D(fake_data), torch.ones(batch_size, 1))
            G.zero_grad();
            D.zero_grad()
            D_loss.backward();
            G_loss.backward()

            for p in list(G.parameters()) + list(D.parameters()):
                p.data.add_(-2 * lr, p.grad.data)
                p.data.add_(lr, p.prev_grad)
                p.prev_grad = p.grad.data.clone()

    return G



# 4. Визуализация результатов 

def plot_results(G_dict):
    real_samples = get_gmm_samples(1000).numpy()
    z = torch.randn(1000, 2)

    # Задаем фиксированный размер фигуры
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (method_name, G) in enumerate(G_dict.items()):
        ax = axes[i]
        fake_samples = G(z).detach().numpy()

        ax.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.4, label='Real Data', c='blue', s=12)
        ax.scatter(fake_samples[:, 0], fake_samples[:, 1], alpha=0.6, label='Generated', c='red', s=12)

       
        ax.set_title(f"Метод: {method_name}", fontsize=13, pad=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("X", fontsize=10, labelpad=5)
        ax.set_ylabel("Y", fontsize=10, labelpad=5)

        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)  # Контроль расстояния по горизонтали между графиками

    plt.savefig('gan_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    generators = {
        'Vanilla GDA': train_toy_gan('GDA', iterations=3000, lr=0.01),
        'Extragradient': train_toy_gan('EG', iterations=3000, lr=0.01),
        'Optimistic GD': train_toy_gan('OGD', iterations=3000, lr=0.01)
    }

    plot_results(generators)
