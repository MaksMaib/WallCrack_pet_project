import matplotlib.pyplot as plt
import torchvision
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from tqdm import tqdm


def train_val_plot(model_name, losses_train, losses_valid):
    plt.ioff()
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, 1 + len(losses_train)), losses_train, 'ro-')
    plt.plot(range(1, 1 + len(losses_valid)), losses_valid, 'bo-')
    plt.xlabel('epoch')
    plt.ylabel('losses [log scale]')
    plt.yscale('log')
    plt.legend(['Train', 'Valid'])
    plt.title(f'{model_name} Train/Val Losses')
    plt.grid()
    plt.savefig(f'outputs/{model_name} Train Validation loss.png')
    plt.close()


def heat_map(images_number, model, loader):
    img, _, _ = next(iter(loader))
    heat_map_loss = nn.MSELoss(reduction='none')
    model.eval()

    img_cuda = img.cuda()

    with torch.no_grad():
        out_data = model(img_cuda)
        heat_map = heat_map_loss(out_data, img_cuda)

    img = img.numpy()
    out_cpu = out_data.cpu().numpy().transpose((0, 2, 3, 1))
    heat_map = heat_map.cpu().numpy().transpose((0, 2, 3, 1))

    for i in range(images_number):
        heat_map_plot = heat_map[i]
        out_plot = out_cpu[i]

        plt.ioff()
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        axs[0].imshow(img[i][0], cmap='gray')
        axs[0].set_title('original')
        axs[1].imshow(out_plot[:, :, 0], cmap='gray')
        axs[1].set_title(f'restored {model._get_name()} ')

        plot_heat = axs[2].imshow(heat_map_plot[:, :, 0], cmap='seismic')
        axs[2].set_title(f'H_map {model._get_name()} ')

        fig.colorbar(plot_heat, ax=axs[2], fraction=0.046, pad=0.01)

        plt.savefig(f'outputs/heat_map/{i} {model._get_name()} heatmap.png')
        plt.close()


def loss_distribution(model, loader, threshold):
    true_labels = []
    negative_img_losses = []
    positive_img_losses = []
    criterion = nn.MSELoss()

    loop = tqdm(loader, total=len(loader), leave=False)  # Iterate over data.
    loop.set_description(f"Loss Distribution {model._get_name()}")
    for data in loop:
        image, label = data[0].cuda(0), data[1]
        with torch.no_grad():
            out_data = model(image)
            for i, iter_img in enumerate(image):
                loss = criterion(iter_img, out_data[i])
                true_labels.append(label[i].item())

                if label[i].item() == 0:
                    negative_img_losses.append(loss.item())

                if label[i].item() == 1:
                    positive_img_losses.append(loss.item())

    plt.ioff()
    plt.figure(figsize=(12, 6))
    plt.title(f'{model._get_name()} Loss Distribution | threshold =  {round(threshold, 6)}')
    sns.histplot(negative_img_losses, bins=250, kde=True, color='blue', label='Label1')
    sns.histplot(positive_img_losses, bins=250, kde=True, color='red', label='Label2')
    plt.axvline(threshold, 0, 10, color='yellow', label='Label3')
    plt.legend(labels=['Negative', 'Positive', 'threshold'])
    plt.savefig(f'outputs/{model._get_name()} loss distribution.png')
    plt.close()
