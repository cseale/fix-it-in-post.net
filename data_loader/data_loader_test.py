import os
import json
import argparse
import data_loader.data_loaders as module_data


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config):
    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()
    print("Data loader length:", len(data_loader))
    print("Validation Data loader length:", len(valid_data_loader))
    print("Batch size:", data_loader.batch_size)
    print("N samples:", data_loader.n_samples)
    print("--------INSPECTING ELEMENTS OF DATA LOADER------")
    for batch_idx, (data, target) in enumerate(data_loader):
        if(batch_idx == 1):
            print("Size of predictors", data.data.numpy().shape)
            print("Size of target", target.data.numpy().shape)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")


    main(config)
