import os
import torchvision
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import pytorch_lightning as pl

random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE=128
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)

class MNISTDataModule(pl.LightningDataModule):
  def __init__(self, data_dir='./', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

    self.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),

        ]
    )

  def prepare_data(self):
    MNIST(self.data_dir, train=True, download=True)
    MNIST(self.data_dir, train=False, download=True)

  def setup(self, stage=None):
    if stage == "fit" or stage is None:
      mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
      self.mnist_train, self.mnist_val = random_split(mnist_full, [55000,5000])

    if stage == "test" or stage is None:
      self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

class Defective(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    # Changed the output size of fc2 to 1 for binary classification
    self.fc2 = nn.Linear(50, 1)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    # Apply sigmoid to get probability between 0 and 1
    return torch.sigmoid(x)

# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]


    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  #256

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Convolution to 28x28 (1 feature map)
        # Use torch.tanh to ensure output is in the range [-1, 1]
        return torch.tanh(self.conv(x))

import torchmetrics

class GAN_Model(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.002):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = VGG16Discriminator()
        self.validation_z = torch.randn(64, self.hparams.latent_dim)
        self.automatic_optimization = False

        # Metrics for accuracy
        self.train_acc = torchmetrics.Accuracy(task=
                                               "binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
      lr = self.hparams.lr
      opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
      opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
      return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        opt_g, opt_d = self.optimizers()

        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim).type_as(real_imgs)

        # Train generator
        fake_imgs = self(z)
        y_hat_gen = self.discriminator(fake_imgs)
        y_gen = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)
        g_loss = self.adversarial_loss(y_hat_gen, y_gen)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log('g_loss', g_loss, prog_bar=True)

        # Train discriminator
        y_hat_real = self.discriminator(real_imgs)
        y_real = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)
        loss_real = self.adversarial_loss(y_hat_real, y_real)

        y_hat_fake = self.discriminator(fake_imgs.detach())
        y_fake = torch.zeros(real_imgs.size(0), 1).type_as(real_imgs)
        loss_fake = self.adversarial_loss(y_hat_fake, y_fake)

        d_loss = (loss_real + loss_fake) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log('d_loss', d_loss, prog_bar=True)

        # Calculate and log accuracy
        preds = torch.cat([y_hat_real, y_hat_fake], dim=0) > 0.5
        targets = torch.cat([y_real, y_fake], dim=0)
        acc = self.train_acc(preds, targets)
        self.log('train_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim).type_as(real_imgs)
        fake_imgs = self(z)

        y_hat_real = self.discriminator(real_imgs)
        y_real = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)

        y_hat_fake = self.discriminator(fake_imgs.detach())
        y_fake = torch.zeros(real_imgs.size(0), 1).type_as(real_imgs)

        preds = torch.cat([y_hat_real, y_hat_fake], dim=0) > 0.5
        targets = torch.cat([y_real, y_fake], dim=0)
        val_acc = self.val_acc(preds, targets)
        self.log('val_acc', val_acc, prog_bar=True)

    def plot_imgs(self):
      z = self.validation_z.type_as(self.generator.lin1.weight)
      sample_imgs = self(z).cpu()
      print("epoch ", self.current_epoch)
      fig = plt.figure(figsize=(8, 6)) # Adjust figure size for better visualization
      num_images = sample_imgs.size(0)
      num_rows = 8  # Number of rows in the grid
      num_cols = 8  # Number of columns in the grid
      for i in range(num_images):
          # Calculate subplot index using num_rows and num_cols
          plt.subplot(num_rows, num_cols, i + 1)
          plt.tight_layout()
          plt.imshow(sample_imgs.detach().numpy()[i, 0, :, :], cmap='gray_r')
          plt.title("Generated Data")
          plt.xticks([])
          plt.yticks([])
          plt.axis('off')
      plt.show()

md = MNISTDataModule()
model = GAN_Model()

model.plot_imgs()

trainer = pl.Trainer(max_epochs=2, devices=AVAIL_GPUS)
trainer.fit(model, md)

import torchvision.models as models

class VGG16Discriminator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load the VGG16 model
        self.vgg16 = models.vgg16(pretrained=pretrained)

        # Modify the first convolutional layer to accept 1 channel input
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Avoid in-place modifications by creating a copy of features
        features = list(self.vgg16.features)

        # ... (Rest of the code for modifying features remains unchanged) ...
        for i, layer in enumerate(self.vgg16.features):
         if isinstance(layer, nn.Conv2d):
         # Modify padding
          layer.padding = (2, 2) # Example: add padding of 2 on each side
          self.vgg16.features = nn.Sequential(*features)

        # Adjust the classifier to output a single value for binary classification
        self.vgg16.classifier[-1] = nn.Linear(in_features=4096, out_features=1)

    def forward(self, x):
        # Ensure input has 3 channels (convert grayscale to RGB if necessary)
        # Removed - This is handled by modifying the first layer
        # if x.shape[1] == 1:
        #    x = x.repeat(1, 3, 1, 1)
        # Avoid in-place sigmoid by using F.sigmoid
        return torch.sigmoid(self.vgg16(x))
