from torchsummary import summary
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torch.optim import lr_scheduler
from os.path import exists
from sklearn.metrics import accuracy_score, recall_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mp_image

from tqdm import tqdm
from IPython.display import display, clear_output


class AutoencoderFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=4, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.Flatten(),

        )

        self.linear = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.decoder(x)
        return x


class AutoencoderCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                                    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=4, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.AvgPool2d(2),
                                     )
        self.decoder = nn.Sequential(
                                    nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0, bias=False),
                                    nn.Tanh(),
                                    )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def data_load_preprocess(path, batch_size=16, percent=0.0, plot_flag=False):

    percent_valid = 1.0  # as 0.05 = 5%

    transform = torchvision.transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,))
    ])

    dataset = ImageFolderWithPaths(path, transform=transform)  # our custom dataset

    # data set's classes equals so: positive data Subset range(0, len(dataset)/2)
    #                               negative data Subset range(len(dataset)/2, len(dataset))
    # that's why 60% of data set = : range(0, int( 0.6 * (len(dataset)/2)))
    #                               range(len(dataset)/2), int( 0.6 * (len(dataset)/2)))

    next_clas_thr = int(len(dataset) * 0.5)

    mask_train = list(np.arange(int(0.6 * next_clas_thr))) + \
                 list(np.arange(next_clas_thr, (int((1 + (0.6 * percent)) * next_clas_thr))))

    mask_valid = list(np.arange((int(0.6 * next_clas_thr)), (int(0.8 * next_clas_thr)))) + \
                 list(np.arange((int(1.6 * next_clas_thr)), (int((1 + (0.8 * percent_valid)) * next_clas_thr))))

    mask_test = list(np.arange((int(0.8 * next_clas_thr)), next_clas_thr)) + \
                list(np.arange((int(1.8 * next_clas_thr)), len(dataset)))

    train_dataset = Subset(dataset, mask_train)
    valid_dataset = Subset(dataset, mask_valid)
    test_dataset = Subset(dataset, mask_test)

    print(f'train dataset len negative/positive = {len(train_dataset)}')
    print(f'valid dataset len negative/positive = {len(valid_dataset)}')
    print(f'test dataset len negative/positive = {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if plot_flag:

        my_imshow(train_loader, title='Train')
        my_imshow(valid_loader, title='Valid')
        my_imshow(test_loader, title='Test')

    return train_loader, valid_loader, test_loader


def my_imshow(img_loader, title=None):
    inputs, _, _ = next(iter(img_loader))
    inputs = torchvision.utils.make_grid(inputs)
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.imshow(inputs)
    plt.show()


def threshold_detection(model):
    all_loss = []
    true_labels = []
    negative_img_losses = []
    positive_img_losses = []
    criterion = nn.MSELoss()
    loop = tqdm(valid_loader, total=len(valid_loader), leave=False)
    loop.set_description(f"Counting loss")
    for data in loop:
        image, label, fname = data[0].to(device), data[1], data[2]
        with torch.no_grad():
            out_data = model(image)
            for i, iter_img in enumerate(image):
                loss = criterion(iter_img, out_data[i])

                if label[i].item() == 0:
                    negative_img_losses.append(loss.item())

                if label[i].item() == 1:
                    positive_img_losses.append(loss.item())

                true_labels.append(label[i].item())
                all_loss.append(loss.item())

    max_acc = 0.0
    threshold = 0.0
    max_rec = 0.0
    num_steps = int(max(all_loss) / 0.00001)

    loop = tqdm(range(0, num_steps), total=num_steps, leave=False)
    loop.set_description(f"Threshold fitting")
    #for i in range(0, num_steps):
    for i in loop:
        predicted_labels = np.array(all_loss)
        predicted_labels[predicted_labels <= i * 0.00001] = 0
        predicted_labels[predicted_labels > i * 0.00001] = 1
        if max_acc < accuracy_score(true_labels, predicted_labels):
            max_acc = accuracy_score(true_labels, predicted_labels)
            max_rec = recall_score(true_labels, predicted_labels)
            threshold = i * 0.00001

        predicted_labels = np.array(all_loss)
        predicted_labels[predicted_labels <= threshold] = 0
        predicted_labels[predicted_labels > threshold] = 1
    plt.ioff()
    plt.figure(figsize=(12, 6))
    plt.title(f'{model._get_name()} Validation Loss Distribution, threshold = {threshold}')
    sns.histplot(negative_img_losses, bins=100, kde=True, color='blue', label='Label1')
    sns.histplot(positive_img_losses, bins=100, kde=True, color='red', label='Label2')
    plt.legend(labels=['Negative', 'Positive'])
    plt.axvline(threshold, 0, 10, color='yellow')
    plt.savefig(f'output_figures/{model._get_name()} Validation loss.png')
    plt.close()
    print(f'{PERCENT * 100}% Cracked images  accuracy = ', max_acc)
    print(f'{PERCENT * 100}% Cracked images recall = ', max_rec)

    return threshold


def loss_distribution(model, threshold):
    max_pix_loss: float = 0.0
    all_loss = []
    true_labels = []
    files_name_list = []
    negative_img_losses = []
    positive_img_losses = []
    criterion = nn.MSELoss()
    criterion_non_red = nn.MSELoss(reduction='none')

    for data in tqdm(test_loader, total=len(test_loader), leave=False):
        image, label, fname = data[0].to(device), data[1], data[2]
        with torch.no_grad():
            out_data = model(image)
            for i, iter_img in enumerate(image):
                loss = criterion(iter_img, out_data[i])
                loss_non_red = criterion_non_red(iter_img, out_data[i])

                if max_pix_loss < loss_non_red.cpu().numpy().max():
                    max_pix_loss = loss_non_red.cpu().numpy().max()

                true_labels.append(label[i].item())

                if label[i].item() == 0:
                    negative_img_losses.append(loss.item())

                if label[i].item() == 1:
                    positive_img_losses.append(loss.item())

                all_loss.append(loss.item())
                files_name_list.append(fname[i])

    predicted_labels = np.array(all_loss)
    predicted_labels[predicted_labels <= threshold] = 0
    predicted_labels[predicted_labels > threshold] = 1

    max_acc = accuracy_score(true_labels, predicted_labels)
    max_rec = recall_score(true_labels, predicted_labels)

    plt.figure(figsize=(12, 6))
    plt.title(f'Loss Distribution, threshold = {threshold}')
    sns.histplot(negative_img_losses, bins=100, kde=True, color='blue', label='Label1')
    sns.histplot(positive_img_losses, bins=100, kde=True, color='red', label='Label2')
    plt.legend(labels=['Negative', 'Positive'])
    plt.axvline(threshold, 0, 10, color='yellow')
    plt.show()

    print(f'{PERCENT * 100}% Cracked images  accuracy = ', max_acc)
    print(f'{PERCENT * 100}% Cracked images recall = ', max_rec)

    miss_classified = []

    wrong_ident = np.where(predicted_labels != true_labels)
    for i in wrong_ident[0]:
        miss_classified.append(files_name_list[i])

    return miss_classified, max_pix_loss


def plot_auto_update(model_name, losses_train, losses_valid):

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
    plt.savefig('output_figures/Validation loss.png')
    plt.close()



def train_loop(model, optim_lr, scheduler_step_size, scheduler_gamma, epoch, safe_model=False):  # envelope for training process

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    epoch_loss_train = []
    epoch_loss_valid = []
    for i in range(epoch):
        loop = tqdm(train_loader, total=len(train_loader), leave=False)  # Iterate over data.
        loop.set_description(f"Training [{i}/{epoch}]")
        running_loss = 0.0
        model.train()

        for data in loop:
            image = data[0].to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, image)
            running_loss += loss.item() * image.size(0)
            loss.backward()
            optimizer.step()
        scheduler.step()
        epoch_loss_train.append(running_loss / len(train_loader))
        # ----------------------------------------------------------------------------------------------------------
        running_loss = 0.0
        model.eval()

        with torch.no_grad():
            loop = tqdm(valid_loader, total=len(valid_loader), leave=False)
            loop.set_description(f"Validation [{i}/{epoch}]")
            for data in loop:
                image = data[0].to(device)
                output = model(image)
                loss = criterion(image, output)
                running_loss += loss.item() * image.size(0)

        epoch_loss_valid.append(running_loss / len(valid_loader))
        plot_auto_update(model._get_name(), epoch_loss_train, epoch_loss_valid)
    if safe_model:
        torch.save(model.state_dict(), 'saved_nn/'+model._get_name())
    return model


def restored_img_mask_plot(images_number, max_loss_flat, max_loss_cnn, model_1, model_2):
    img, _, _ = next(iter(test_loader))
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

        fig, axs = plt.subplots(1, 5, figsize=(18, 8))
        axs[0].imshow(img[i][0], cmap='gray')
        axs[0].set_title('original')

        axs[1].imshow(out_plot1[:, :, 0], cmap='gray')
        axs[1].set_title(f'restored {model_1._get_name()} ')

        axs[2].imshow(out_plot2[:, :, 0], cmap='gray')
        axs[2].set_title(f'restored {model_2._get_name()} ')

        axs[3].imshow(heat_map_plot1[:, :, 0], cmap='seismic', vmin=0, vmax=max(max_loss_cnn, max_loss_flat))
        axs[3].set_title(f'H_map {model_1._get_name()} ')

        plot_heat_4 = axs[4].imshow(heat_map_plot2[:, :, 0], cmap='seismic', vmin=0, vmax=max(max_loss_cnn,
                                                                                              max_loss_flat)
                                    )
        axs[4].set_title(f'H_map {model_2._get_name()} ')
        fig.colorbar(plot_heat_4, ax=axs[4], fraction=0.046, pad=0.01)

        plt.show()


if __name__ == '__main__':
    PATH = "Concrete Crack Images for Classification/"
    BATCH_SIZE: int = 16
    PERCENT = 0.0
    num_epochs = 150
    learning_rate = 0.0001
    scheduler_step_size = 10
    scheduler_gamma = 0.5

    try_to_load_pretrain = False
    nn_name_cnn = "Autoencoedr_cnn_0605.pt"
    nn_name_flatten = "Autoencoedr_flatten_0605.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    train_loader, valid_loader, test_loader = data_load_preprocess(
                                                                  path=PATH, batch_size=BATCH_SIZE,
                                                                  percent=PERCENT, plot_flag=False
                                                                  )

    if try_to_load_pretrain and exists("saved_nn/" + nn_name_cnn):
        AutoencoderTrainCnn = AutoencoderCnn()
        AutoencoderTrainCnn.load_state_dict(torch.load("saved_nn/" + nn_name_cnn))
        AutoencoderTrainCnn.to(device).eval()
    else:
        AutoencoderTrainCnn = AutoencoderCnn().to(device)
        AutoencoderTrainCnn = train_loop(AutoencoderTrainCnn, learning_rate, scheduler_step_size,
                                         scheduler_gamma,
                                         num_epochs,
                                         safe_model=True
                                         )

    if try_to_load_pretrain and exists("saved_nn/" + nn_name_flatten):
        AutoencoderTrainFlatten = AutoencoderFlatten()
        AutoencoderTrainFlatten.load_state_dict(torch.load("saved_nn/" + nn_name_flatten))
        AutoencoderTrainFlatten.to(device).eval()

    else:
        AutoencoderTrainFlatten = AutoencoderFlatten().to(device)
        AutoencoderTrainFlatten = train_loop(AutoencoderTrainFlatten, learning_rate, scheduler_step_size,
                                             scheduler_gamma,
                                             num_epochs,
                                             safe_model=True
                                             )

    threshold_cnn= threshold_detection(AutoencoderTrainCnn)
    threshold_flatten = threshold_detection(AutoencoderTrainFlatten)
    print('end',threshold_flatten, threshold_cnn)