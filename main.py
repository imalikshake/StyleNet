import argparse
import tensorflow as tf
import os
import numpy as np
from model import GenreLSTM


parser = argparse.ArgumentParser(description='How to run this')

parser.add_argument(
    "-current_run",
    type=str,
    help="The name of the model which will also be the name of the session's folder."
)

parser.add_argument(
    "-data_dir",
    type=str,
    default="./data",
    help="Directory of datasets"
)

parser.add_argument(
    "-data_set",
    type=str,
    default="test",
    help="The name of training dataset"
)

parser.add_argument(
    "-runs_dir",
    type=str,
    default="./runs",
    help="The name of the model which will also be the name of the session folder"
)

parser.add_argument(
    "-bi",
    help="True for bidirectional",
    action='store_true'
)

parser.add_argument(
    "-forward_only",
    action='store_true',
    help="True for forward only, False for training [False]"
)

parser.add_argument(
    "-load_model",
    type=str,
    default=None,
    help="Folder name of model to load"
)

parser.add_argument(
    "-load_last",
    action='store_true',
    help="Start from last epoch"
)

args = parser.parse_args()

def setup_dir():

    print('[*] Setting up directory...')

    main_path = args.runs_dir
    current_run = os.path.join(main_path, args.current_run)

    files_path = args.data_dir
    files_path = os.path.join(files_path, args.data_set)

    x_path = os.path.join(files_path, 'inputs')
    y_path = os.path.join(files_path, 'velocities')
    eval_path = os.path.join(files_path, 'eval')

    model_path = os.path.join(current_run, 'model')
    logs_path = os.path.join(current_run, 'tmp')
    png_path = os.path.join(current_run, 'png')
    pred_path = os.path.join(current_run, 'predictions')

    if not os.path.exists(current_run):
        os.makedirs(current_run)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    dirs = {
            'main_path': main_path,
            'current_run': current_run,
            'model_path': model_path,
            'logs_path': logs_path,
            'png_path': png_path,
            'eval_path': eval_path,
            'pred_path': pred_path,
            'x_path': x_path,
            'y_path': y_path
        }

    # print main_path
    # print current_run
    # print model_path
    # print logs_path
    # print png_path
    # print eval_path
    # print x_path
    # print y_path
    return dirs

def load_training_data(x_path, y_path, genre):
    X_data = []
    Y_data = []
    names = []
    print('[*] Loading data...')

    x_path = os.path.join(x_path, genre)
    y_path = os.path.join(y_path, genre)

    for i, filename in enumerate(os.listdir(x_path)):
        if filename.split('.')[-1] == 'npy':
            names.append(filename)

    for i, filename in enumerate(names):
        abs_x_path = os.path.join(x_path,filename)
        abs_y_path = os.path.join(y_path,filename)
        loaded_x = np.load(abs_x_path)

        X_data.append(loaded_x)

        loaded_y = np.load(abs_y_path)
        loaded_y = loaded_y/127
        Y_data.append(loaded_y)
        assert X_data[i].shape[0] == Y_data[i].shape[0]


    return X_data, Y_data

def prepare_data():
    dirs = setup_dir()
    data = {}
    data["classical"] = {}
    data["jazz"] = {}

    c_train_X , c_train_Y = load_training_data(dirs['x_path'], dirs['y_path'], "classical")

    data["classical"]["X"] = c_train_X
    data["classical"]["Y"] = c_train_Y

    j_train_X , j_train_Y = load_training_data(dirs['x_path'], dirs['y_path'], "jazz")

    data["jazz"]["X"] = j_train_X
    data["jazz"]["Y"] = j_train_Y
    return dirs, data

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)

    dirs, data = prepare_data()

    network  = GenreLSTM(dirs, input_size=176, mini=True, bi=args.bi)
    network.prepare_model()

    if not args.forward_only:
        if args.load_model:
            loaded_epoch = args.load_model.split('.')[0]
            loaded_epoch = loaded_epoch.split('-')[-1]
            loaded_epoch = loaded_epoch[1:]
            print("[*] Loading " + args.load_model + " and continuing from " + loaded_epoch + ".")
            loaded_epoch = int(loaded_epoch)
            network.train(data, model=args.load_model, starting_epoch=loaded_epoch+1)
        elif args.load_last:
            tree = os.listdir(dirs["model_path"])
            tree.remove('checkpoint')
            files = [(int(file.split('.')[0].split('-')[-1][1:]), file.split('.')[0]) for file in tree]
            files.sort(key = lambda t: t[0])
            # print files
            last = files[-1][1]
            last = last + ".ckpt"
            loaded_epoch = files[-1][0]
            # loaded_epoch = last.split('-')[-1]
            # loaded_epoch = loaded_epoch[1:]
            # last = last + ".ckpt"
            print("[*] Loading " + last + " and continuing from " + str(loaded_epoch) + ".")
            network.train(data, model=last, starting_epoch=loaded_epoch+1)
        else:
            network.train(data)
    else:
        network.load(args.load_model)

if __name__ == '__main__':
    main()
