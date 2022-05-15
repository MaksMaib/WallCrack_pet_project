# coding:utf8
import warnings
from torchvision import transforms


class DefaultConfig(object):
    #PATH = "Concrete Crack Images for Classification/"
    PATH = "Train/"
    BATCH_SIZE = 16
    PERCENT = 1.0
    num_epoc = 150
    learning_rate = 0.0001
    scheduler_step_size = 10
    scheduler_gamma = 0.5
    resize_img = 64
    try_to_load_pretrain = True
    model_name = 'AutoencoderCnn'

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute {}".format(k))
                setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print (k, getattr(self, k))


opt = DefaultConfig()
