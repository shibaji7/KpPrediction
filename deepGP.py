"""
GP-LSTM regression on Actuator data.
"""
from __future__ import print_function

import os
import numpy as np
np.random.seed(42)

# Keras
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

# Dataset interfaces
from kgp.datasets.sysid import load_data
from kgp.datasets.data_utils import data_to_seq, standardize_data

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.experiment import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE

os.environ["GPML_PATH"] = "/home/shibaji7/anaconda3/envs/deep/lib/python2.7/site-packages/kgp/backend/gpml/"
os.system("module load matlab")

def main():
    # Load data
    X, y = load_data('drives', use_targets=False)
    print(X.shape,y.shape)
    X_seq, y_seq = data_to_seq(X, y,
        t_lag=32, t_future_shift=1, t_future_steps=1, t_sw_step=1)
    
    print(X_seq.shape,y_seq.shape)
    # Split
    train_end = int((45. / 100.) * len(X_seq))
    test_end = int((90. / 100.) * len(X_seq))
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_test, y_test = X_seq[train_end:test_end], y_seq[train_end:test_end]
    X_valid, y_valid = X_seq[test_end:], y_seq[test_end:]

    data = {
        'train': [X_train, y_train],
        #'valid': [X_valid, y_valid],
        'test': [X_test, y_test],
    }

    # Re-format targets
    for set_name in data:
        y = data[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data[set_name][1] = [y[:,:,i] for i in xrange(y.shape[2])]
    print(data["train"][0].shape, np.array(data["train"][1]).shape)
    # Model & training parameters
    nb_train_samples = data['train'][0].shape[0]
    input_shape = list(data['train'][0].shape[1:])
    nb_outputs = len(data['train'][1])
    gp_input_shape = (1,)
    batch_size = 128
    epochs = 5
    print(input_shape)

    nn_params = {
        'H_dim': 16,
        'H_activation': 'tanh',
        'dropout': 0.1,
    }
    gp_params = {
        'cov': 'SEiso',
        'hyp_lik': -2.0,
        'hyp_cov': [[-0.7], [0.0]],
        'opt': {},
    }

    # Retrieve model config
    nn_configs = load_NN_configs(filename='lstm.yaml',
                                 input_shape=input_shape,
                                 output_shape=gp_input_shape,
                                 params=nn_params)
    gp_configs = load_GP_configs(filename='gp.yaml',
                                 nb_outputs=nb_outputs,
                                 batch_size=batch_size,
                                 nb_train_samples=nb_train_samples,
                                 params=gp_params)

    # Construct & compile the model
    model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['GP']])
    loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
    model.compile(optimizer=Adam(1e-2), loss=loss)

    # Callbacks
    callbacks = [EarlyStopping(monitor='mse', patience=10)]

    # Train the model
    history = train(model, data, callbacks=callbacks, gp_n_iter=5,
                    checkpoint='lstm', checkpoint_monitor='mse',
                    epochs=epochs, batch_size=batch_size, verbose=2)

    # Finetune the model
    model.finetune(*data['train'],
                   batch_size=batch_size,
                   gp_n_iter=100,
                   verbose=0)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test predict RMSE:', rmse_predict)


if __name__ == '__main__':
    main()

