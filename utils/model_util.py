import torch
from model.baseline_model import FullyConnectedBaseline as fcnetwork, ConvolutionalBaseline_TimeFiltering as conv_time


def load_model(n_features=129, n_segments=8, model_to_test="Baseline_FullyConnected/0505_130215", type="fc"):
    model = create_model(type, n_features, n_segments)
    model_path = "./saved/" + model_to_test + "/model_best.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])

    return model


def create_model(type, n_features, n_segments):
    if type == "fc":
        return create_fc_model(n_features, n_segments)
    if type == 'conv_time':
        return create_conv_time_model(n_features, n_segments)
        
    # TODO: with every model we should create new model function

    return create_fc_model(n_features, n_segments)


def create_fc_model(n_features, n_segments):
    return fcnetwork(n_features=n_features, n_segments=n_segments)

def create_conv_time_model(n_features, n_segments):
    return conv_time(n_features=n_features, n_segments=n_segments)
