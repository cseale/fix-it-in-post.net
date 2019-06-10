import torch
from model.fully_connected import FullyConnectedBaseline as fcnetwork
from model.deep_fully_connected import DeepFullyConnectedBaseline as fcdnetwork
from model.convolutional_shallow import ConvolutionalShallow as convsh
from model.convolutional_deep import ConvolutionalDeep as convd
from model.convolutional_deep_time import ConvolutionalDeep_Time as convd_time
from model.convolutional_shallow_time import ConvolutionalShallow_Time as convsh_time
from model.lstm_v2 import LSTMModel as lstm


def load_model(n_features=129, n_segments=8, model_to_test="Baseline_FullyConnected/0505_130215", type="fc"):
    model = create_model(type, n_features, n_segments)
    model_path = model_to_test + "/model_best.pth"
    model.load_state_dict(torch.load(model_path, 'cpu')['state_dict'])

    return model
 

def create_model(type, n_features, n_segments):
    if type == "fc":
        return create_fc_model(n_features, n_segments)
    if type == "fc_deep":
        return create_fc_deep_model(n_features, n_segments)
    if type == 'conv_shallow':
        return create_conv_shallow_model(n_features, n_segments)
    if type == 'conv_shallow_time':
        return create_conv_shallow_time_model(n_features, n_segments)
    if type == 'conv_deep':
        return create_conv_deep_model(n_features, n_segments)
    if type == 'conv_deep_time':
        return create_conv_deep_time_model(n_features, n_segments)
    if type == 'lstm':
        return create_lstm_model(n_features)
    # TODO: with every model we should create new model function

    return create_fc_model(n_features, n_segments)


def create_fc_model(n_features, n_segments):
    return fcnetwork(n_features=n_features, n_segments=n_segments)


def create_fc_deep_model(n_features, n_segments):
    return fcdnetwork(n_features=n_features, n_segments=n_segments)


def create_conv_shallow_time_model(n_features, n_segments):
    return convsh_time(n_features=n_features, n_segments=n_segments)


def create_conv_deep_time_model(n_features, n_segments):
    return convd_time(n_features=n_features, n_segments=n_segments)


def create_conv_shallow_model(n_features, n_segments):
    return convsh(n_features=n_features, n_segments=n_segments)


def create_conv_deep_model(n_features, n_segments):
    return convd(n_features=n_features, n_segments=n_segments)


def create_lstm_model(n_features):
    return lstm(n_features=n_features, batch_size = 4, use_cuda=False)
