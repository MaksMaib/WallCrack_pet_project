
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader, Subset


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


def data_load_preprocess(path, percent=0.0, resize_img=None, plot_flag=False):

    percent_valid = 1.0  # as 0.05 = 5%
    transform = transforms.Compose([
        transforms.Resize(resize_img),
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

    return train_dataset, valid_dataset, test_dataset
