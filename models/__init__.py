from models.AutoencoderCnn import AutoencoderCnn
from models.AutoencoderFlatten import AutoencoderFlatten


def get_model(name):
    model = _get_model_instance(name)

    if name == 'AutoencoderCnn':
        model = model()
    elif name == 'AutoencoderFlatten':
        model = model()
    else:
        raise 'Model {} not available'.format(name)
    return model

def _get_model_instance(name):
    return {
            'AutoencoderCnn':AutoencoderCnn,
            'AutoencoderFlatten':AutoencoderFlatten,

        }[name]