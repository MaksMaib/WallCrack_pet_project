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


def test_vis(model, threshold, max_acc, positive_img_losses, negative_img_losses):
    plt.ioff()
    plt.figure(figsize=(12, 6))
    plt.title(
        f'{model._get_name()} Loss Distribution | Accuracy = {round(max_acc * 100, 4)}% | threshold =  {round(threshold, 6)}')
    sns.histplot(negative_img_losses, bins=100, kde=True, color='blue', label='Label1')
    sns.histplot(positive_img_losses, bins=100, kde=True, color='red', label='Label2')
    plt.legend(labels=['Negative', 'Positive'])
    plt.axvline(threshold, 0, 10, color='yellow')
    plt.savefig(f'output_figures/{model._get_name()} Test Loss Distribution.png')
    plt.close()


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


def heat_map(images_number, max_loss_flat, max_loss_cnn, model_1, model_2, loader):
    img, _, _ = next(iter(loader))
    heat_map_loss = nn.MSELoss(reduction='none')
    model_1.eval()
    model_2.eval()
    img_cuda = img.cuda()

    with torch.no_grad():
        out_data1 = model_1(img_cuda)
        out_data2 = model_2(img_cuda)

        heat_map1 = heat_map_loss(out_data1, img_cuda)
        heat_map2 = heat_map_loss(out_data2, img_cuda)

    img = img.numpy()
    out_cpu1 = out_data1.cpu().numpy().transpose((0, 2, 3, 1))
    out_cpu2 = out_data2.cpu().numpy().transpose((0, 2, 3, 1))

    heat_map1 = heat_map1.cpu().numpy().transpose((0, 2, 3, 1))
    heat_map2 = heat_map2.cpu().numpy().transpose((0, 2, 3, 1))

    for i in range(images_number):
        heat_map_plot1 = heat_map1[i]
        heat_map_plot2 = heat_map2[i]

        out_plot1 = out_cpu1[i]
        out_plot2 = out_cpu2[i]
        plt.ioff()
        fig, axs = plt.subplots(1, 5, figsize=(18, 8))
        axs[0].imshow(img[i][0], cmap='gray')
        axs[0].set_title('original')

        axs[1].imshow(out_plot1[:, :, 0], cmap='gray')
        axs[1].set_title(f'restored {model_1._get_name()} ')

        axs[2].imshow(out_plot2[:, :, 0], cmap='gray')
        axs[2].set_title(f'restored {model_2._get_name()} ')

        axs[3].imshow(heat_map_plot1[:, :, 0], cmap='seismic',
                      vmin=0, vmax=max(max_loss_cnn, max_loss_flat))

        axs[3].set_title(f'H_map {model_1._get_name()} ')

        plot_heat_4 = axs[4].imshow(heat_map_plot2[:, :, 0], cmap='seismic',
                                    vmin=0, vmax=max(max_loss_cnn, max_loss_flat))

        axs[4].set_title(f'H_map {model_2._get_name()} ')
        fig.colorbar(plot_heat_4, ax=axs[4], fraction=0.046, pad=0.01)

        plt.savefig(f'heat_map/{i}heatmap.png')
        plt.close()


def wrong_predicted_img(miss_classified, max_loss, model):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    for i, img_path in enumerate(miss_classified):
        image = Image.open(img_path)
        img = transform(image)
        img = np.expand_dims(img, axis=0)
        img = torch.Tensor(img)
        loss_func = nn.MSELoss()
        heat_map_loss = nn.MSELoss(reduction='none')
        model.eval()
        img_cuda = img.cuda()
        with torch.no_grad():
            out_data = model(img_cuda)
            avg_loss = loss_func(out_data, img_cuda).item()
            pixels_loss = heat_map_loss(out_data, img_cuda)

        img = img.numpy()

        out_cpu = out_data.cpu().numpy().transpose((0, 2, 3, 1))
        pixels_loss = pixels_loss[0].cpu().numpy().transpose((1, 2, 0))

        img = img[0] * 1.0 / img.max()
        plt.ioff()
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        axs[0].imshow(img.transpose((1, 2, 0)), cmap='gray')
        axs[0].set_title('original')

        axs[1].imshow(out_cpu[0], cmap='gray')
        axs[1].set_title(f'restored ')

        plot_heat_1 = axs[2].imshow(pixels_loss[:, :, 0], cmap='seismic', vmin=0, vmax=max_loss)
        axs[2].set_title(f'H_map  {avg_loss}')
        fig.colorbar(plot_heat_1, ax=axs[2], fraction=0.046, pad=0.04)
        plt.suptitle("Examples of misclassified images", fontsize=20)
        plt.savefig(f'wrong_predicted_img/{model._get_name()}_{i}_heatmap.png')
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