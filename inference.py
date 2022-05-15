import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from os.path import exists
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

from config import opt
from models import get_model
from loader import crack_wall_loader
import visualization


def test(**kwargs):
    opt.parse(kwargs)
    model = get_model(opt.model_name)
    if exists("saved_nn/" + model._get_name() + '.pt'):
        model.load_state_dict(torch.load("saved_nn/" + model._get_name() + '.pt'))
        model.cuda(0).eval()
    else:
        raise 'Trained model not found'

    with open(f"saved_nn/threshold for {model._get_name()}.txt") as f:
        threshold = float(f.read())

    _, _, test_dataset = crack_wall_loader.data_load_preprocess(
                        path=opt.PATH, percent=opt.PERCENT, resize_img=opt.resize_img
                                                                )

    test_loader = DataLoader(test_dataset, batch_size=opt.BATCH_SIZE, shuffle=True)

    all_loss = []
    true_labels = []
    files_name_list = []

    criterion = nn.MSELoss()

    loop = tqdm(test_loader, total=len(test_loader), leave=False)  # Iterate over data.
    loop.set_description(f"Loss Distribution {model._get_name()}")
    for data in loop:
        image, label, fname = data[0].cuda(0), data[1], data[2]
        with torch.no_grad():
            out_data = model(image)
            for i, iter_img in enumerate(image):
                loss = criterion(iter_img, out_data[i])
                true_labels.append(label[i].item())
                all_loss.append(loss.item())
                files_name_list.append(fname[i])

    predicted_labels = np.array(all_loss)
    predicted_labels[predicted_labels <= threshold] = 0
    predicted_labels[predicted_labels > threshold] = 1

    max_acc = accuracy_score(true_labels, predicted_labels)
    print(f'{model._get_name()} accuracy = ', max_acc)


def heatmap_plot(images_number, **kwargs):
    opt.parse(kwargs)
    model = get_model(opt.model_name)
    if exists("saved_nn/" + model._get_name() + '.pt'):
        model.load_state_dict(torch.load("saved_nn/" + model._get_name() + '.pt'))
        model.cuda(0).eval()
    else:
        raise 'Trained model not found'
    with open(f"saved_nn/threshold for {model._get_name()}.txt") as f:
        threshold = float(f.read())

    _, _, test_dataset = crack_wall_loader.data_load_preprocess(
        path=opt.PATH, percent=opt.PERCENT, resize_img=opt.resize_img
    )

    test_loader = DataLoader(test_dataset, batch_size=opt.BATCH_SIZE, shuffle=True)

    visualization.heat_map(images_number, model, test_loader)


def loss_distrib(**kwargs):
    opt.parse(kwargs)
    model = get_model(opt.model_name)
    if exists("saved_nn/" + model._get_name() + '.pt'):
        model.load_state_dict(torch.load("saved_nn/" + model._get_name() + '.pt'))
        model.cuda(0).eval()
    else:
        raise 'Trained model not found'

    with open(f"saved_nn/threshold for {model._get_name()}.txt") as f:
        threshold = float(f.read())

    _, _, test_dataset = crack_wall_loader.data_load_preprocess(
                        path=opt.PATH, percent=opt.PERCENT, resize_img=opt.resize_img
                                                                )

    test_loader = DataLoader(test_dataset, batch_size=opt.BATCH_SIZE, shuffle=True)

    visualization.loss_distribution(model, test_loader, threshold)


if __name__ == '__main__':
    import fire
    fire.Fire()
