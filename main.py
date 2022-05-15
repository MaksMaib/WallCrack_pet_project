import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from os.path import exists
from sklearn.metrics import accuracy_score, recall_score
import numpy as np

from tqdm import tqdm
from config import opt
from models import get_model
from loader import crack_wall_loader
import visualization


def _threshold_detection(model, loader):
    all_loss = []
    true_labels = []
    criterion = nn.MSELoss()
    loop = tqdm(loader, total=len(loader), leave=False)
    loop.set_description(f"Counting loss...")
    for data in loop:
        image, label = data[0].cuda(0), data[1]
        with torch.no_grad():
            out_data = model(image)
            for i, iter_img in enumerate(image):
                loss = criterion(iter_img, out_data[i])
                true_labels.append(label[i].item())
                all_loss.append(loss.item())

    max_acc = 0.0
    threshold = 0.0
    max_rec = 0.0
    num_steps = int(max(all_loss) / 0.00001)
    loop = tqdm(range(0, num_steps), total=num_steps, leave=False)
    loop.set_description(f"Threshold fitting")
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
    print(f'{model._get_name()}% accuracy = ', max_acc, '\n')
    return threshold


def train(**kwargs):
    opt.parse(kwargs)
    model = get_model(opt.model_name)
    model.cuda(0)

    train_dataset, valid_dataset, _ = crack_wall_loader.data_load_preprocess(
                        path=opt.PATH, percent=opt.PERCENT, resize_img=opt.resize_img, plot_flag=False
                                                                            )

    train_loader = DataLoader(train_dataset, batch_size=opt.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.BATCH_SIZE, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_step_size,
                                                gamma=opt.scheduler_gamma)

    epoch_loss_train = []
    epoch_loss_valid = []
    for i in range(opt.num_epoch):
        loop = tqdm(train_loader, total=len(train_loader), leave=False)  # Iterate over data.
        loop.set_description(f"Training [{i}/{opt.num_epoch}]")
        running_loss = 0.0
        model.train()
        for data in loop:
            image = data[0].cuda(0)
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
            loop.set_description(f"Validation [{i}/{opt.num_epoch}]")
            for data in loop:
                image = data[0].cuda(0)
                output = model(image)
                loss = criterion(image, output)
                running_loss += loss.item() * image.size(0)
        epoch_loss_valid.append(running_loss / len(valid_loader))

        if opt.train_val_visual:
            visualization.train_val_plot(model._get_name(), epoch_loss_train, epoch_loss_valid)

    threshold = _threshold_detection(model, valid_loader)
    with open(f"'saved_nn/threshold for {model._get_name()}.txt", "w") as output:
        output.write((str(threshold)))
    torch.save(model.state_dict(), 'saved_nn/' + model._get_name() + '.pt')


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

    test_loader = DataLoader(test_dataset, batch_size=opt.BATCH_SIZE, shuffle=False)

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
    max_rec = recall_score(true_labels, predicted_labels)
    print(f'{model._get_name()} accuracy = ', max_acc)


if __name__ == '__main__':
    import fire
    fire.Fire()