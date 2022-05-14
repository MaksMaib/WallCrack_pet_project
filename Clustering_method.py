from torchsummary import summary
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn import decomposition
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans, Birch

from sklearn import decomposition
from sklearn.decomposition import IncrementalPCA


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp_image

from tqdm import tqdm

from sklearn.metrics import accuracy_score


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

    transform = torchvision.transforms.Compose([
                                                transforms.ToTensor(),
                                               # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 #                    std=[0.229, 0.224, 0.225])
                                                 ])

    dataset = ImageFolderWithPaths(path, transform=transform)  # our custom dataset
    next_clas_thr = int(len(dataset) * 0.5)

    mask_train = list(np.arange(int(0.8 * next_clas_thr))) + \
                 list(np.arange(next_clas_thr, (int((1 + (0.8 * percent)) * next_clas_thr))))

    mask_test = list(np.arange((int(0.8 * next_clas_thr)), next_clas_thr)) + \
                list(np.arange((int(1.8 * next_clas_thr)), len(dataset)))

    train_dataset = Subset(dataset, mask_train)
    test_dataset = Subset(dataset, mask_test)

    print(f'train dataset len negative/positive = {len(train_dataset)}')
    print(f'test dataset len negative/positive = {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



    if plot_flag:

        my_imshow(train_loader, title='Train')
        my_imshow(test_loader, title='Test')

    return train_loader, test_loader


def my_imshow(img_loader, title=None):
    inputs, _, _ = next(iter(img_loader))
    inputs = torchvision.utils.make_grid(inputs)
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.imshow(inputs)
    plt.show()


def prediction_plot(model, features_2d, labels_ground_truth, labels_predicted):

    print(f'{model}  Accuracy  = ', accuracy_score(labels_ground_truth, labels_predicted))

    wrong_predict = []
    wrong_ind = np.where(labels_predicted != labels_ground_truth)
    for i in wrong_ind[0]:
        wrong_predict.append(features_2d[i])
    wrong_predict = np.array(wrong_predict)

    plt.ioff()
    plt.figure(figsize=(12, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_predicted)
    plt.scatter(wrong_predict[:, 0], wrong_predict[:, 1], color='red', alpha=0.5)
    plt.xlabel('IPC_x')
    plt.ylabel('IPC_y')
    plt.title(f'{model}  prediction | Accuracy = {accuracy_score(labels_ground_truth, labels_predicted)}')
    plt.savefig(f'classic_methods_outputs/{model} prediction.png')
    plt.close()

    return accuracy_score(labels_ground_truth, labels_predicted)


def pca_fit_transform():
    ipca_2D = IncrementalPCA(n_components=2, batch_size=BATCH_SIZE)

    features_2D = []
    labels_ground_truth = []
    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    loop.set_description(f"PCA fitting")
    for images, labels, _ in loop:
        images = images.to(device)
        outputs = resnet(images)
        ipca_2D.partial_fit(outputs.to('cpu').numpy())

    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    loop.set_description(f"PCA transforming to 2D")
    for images, labels, _ in loop:
        labels_ground_truth = np.append(labels_ground_truth, np.array(labels))
        images = images.to(device)
        outputs = resnet(images)
        features_2D.append(ipca_2D.transform(outputs.to('cpu')))

    labels_ground_truth = np.array(labels_ground_truth, dtype=int)

    x_2D = []
    for feature_batch in features_2D:
        for feat in feature_batch:
            x_2D.append(feat)

    x_2D = np.array(x_2D)

   # prediction_plot(ipca_2D,x_2D, labels_ground_truth, )

    return labels_ground_truth, x_2D


def invert(labels):
    inv_labels = []
    for i in range(len(labels)):
        if labels[i]:
            inv_labels = np.append(inv_labels, 0)
        else:
            inv_labels = np.append(inv_labels, 1)
    return inv_labels


def kmeans_fit_predict(labels_ground_truth, features_2d):
    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=BATCH_SIZE)
    labels_predicted = []
    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    loop.set_description(f"K-Means fitting")
    for images, labels, _ in loop:
        images = images.to(device)
        outputs = resnet(images)
        kmeans.partial_fit(outputs.to('cpu').numpy())

    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    loop.set_description(f"K-Means predicting")
    for images, labels, _ in loop:
        images = images.to(device)
        outputs = resnet(images)
        labels_predicted = np.append(labels_predicted, np.array(kmeans.predict(outputs.to('cpu').numpy())))

    if accuracy_score(labels_ground_truth, labels_predicted) < 0.5:
        labels_predicted = invert(labels_predicted)

    accuracy = prediction_plot(kmeans, features_2d,labels_ground_truth, labels_predicted)
    return accuracy


def birch_fit_predict(labels_ground_truth,features_2d):

    brc = Birch(n_clusters=2)
    labels_predicted = []
    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    loop.set_description(f"BIRCH fitting")
    for images, labels, _ in loop:
        images = images.to(device)
        outputs = resnet(images)
        brc.partial_fit(outputs.to('cpu').numpy())

    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    loop.set_description(f"BIRCH predicting")
    for images, labels, _ in loop:
        images = images.to(device)
        outputs = resnet(images)
        labels_predicted = np.append(labels_predicted, np.array(brc.predict(outputs.to('cpu').numpy())))

    if accuracy_score(labels_ground_truth, labels_predicted) < 0.5:
        labels_predicted = invert(labels_predicted)

    accuracy = prediction_plot(brc, features_2d, labels_ground_truth, labels_predicted)
    return accuracy


if __name__ == '__main__':
    #PATH = "Concrete Crack Images for Classification/"
    PATH = "Train/"
    BATCH_SIZE: int = 16
    PERCENT = 1.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet = torchvision.models.resnet18(pretrained=True).to(device)
    resnet.fc = nn.Flatten()
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.eval()

    train_loader, test_loader = data_load_preprocess(
                                                        path=PATH, batch_size=BATCH_SIZE,
                                                        percent=PERCENT, plot_flag=False
                                                    )

    labels_ground_truth, features_2d = pca_fit_transform()
    kmeans_fit_predict(labels_ground_truth, features_2d)
    birch_fit_predict(labels_ground_truth, features_2d)
    print('complete')
