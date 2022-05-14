# coding:utf8
import warnings
from torchvision import transforms


class DefaultConfig(object):
    #   PATH = "Concrete Crack Images for Classification/"
    PATH = "Train/"
    BATCH_SIZE: int = 16
    PERCENT = 1.0
    num_epochs = 150
    learning_rate = 0.0001
    scheduler_step_size = 10
    scheduler_gamma = 0.5

    try_to_load_pretrain = True
    nn_name_cnn = "AutoencoderCnn.pt"
    nn_name_flatten = "AutoencoderFlatten.pt"


opt = DefaultConfig()
