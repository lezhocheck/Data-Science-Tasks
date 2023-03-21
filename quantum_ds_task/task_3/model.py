import pandas as pd
import numpy as np
from argparse import ArgumentParser, FileType, Namespace
import pickle
import os


# path to the directory that contains this python script
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# paths to the trained model and scaler
MODEL_SAVE_FILE_PATH = DIR_PATH + '/regression.save'
SCALER_SAVE_FILE_PATH = DIR_PATH + '/scaler.save'


class Model:
    def __init__(self) -> None:
        with open(MODEL_SAVE_FILE_PATH, 'rb') as model_file:
            self._model = pickle.load(model_file)
        with open(SCALER_SAVE_FILE_PATH, 'rb') as scaler_file:
            self._scaler = pickle.load(scaler_file)
    
    def predict(self, data_path: str) -> np.ndarray:
        input_data = pd.read_csv(data_path)
        scaled_data = self._scaler.transform(input_data)
        return self._model.predict(scaled_data)


def start() -> None:
    args = get_command_arguments()
    model = Model()
    result = model.predict(args.data.name)
    write_predictions(result, args)


def get_command_arguments() -> Namespace:
    parser = ArgumentParser(prog='Regression on tabular data', 
                            description='Model for regression on tabular data task.')
    parser.add_argument('-data', type=FileType('r', encoding='UTF-8'), required=True, 
                        help='A valid path to the input data file.')
    parser.add_argument('--out', type=FileType('w', encoding='UTF-8'), 
                        help='Name of the output file, which will store predictions.')
    return parser.parse_args()


# if --out is not specified writes to standart output
def write_predictions(predictions: np.ndarray, args: Namespace) -> None:
    frame = pd.DataFrame(data=predictions, columns=['predictions'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        if args.out is None:
            print(frame)
        else:
            frame.to_csv(args.out.name)


if __name__ == '__main__':
    start()
