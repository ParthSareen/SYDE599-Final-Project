import DataTransforms
import FogDatasetLoader
import torch
import Model
from torch.utils.data import DataLoader
import optuna

def main():
    fdl = FogDatasetLoader.FogDatasetLoader('./data')
    loader = DataLoader(fdl, batch_size=1, shuffle=False)
    dt = DataTransforms(loader)

def objective(trial):
    #optuna: window size, sequence length, d_model, nhead,n_encoders
    #constant parameters: d_feed_forward: kernel size, d_conv_layers, step size = 1/4 window
    cfg = {     'seq_length': trial.suggest_categorical('seq_length', 256, 512, 2048),
                    'd_model' : trial.suggest_categorical('d_model', 16, 32),
                    'nhead' : trial.suggest_categorical('nhead', 2,4,8),
                    'n_encoders' : trial.suggest_int('n_encoders', 1,3)
            }
    NUM_EPOCH = 5

    d_feed_forward = 512
    d_emg = 10
    d_ecg = 10
    d_imu = 10
    d_skin = 10
    num_init_conv_layers = 3
    num_all_conv_layers = 2
    kernel_size = 8
    d_conv_layers = 32
    encoder_dropout = 0.5
    max_pool_dim = 4
    d_mlp = 128
    n_mlp_layers = 2
    model = Model(seq_length=cfg['seq_length'], d_model=cfg['d_model'], nhead=cfg['nhead'], d_feed_forward, d_emg, d_ecg, max_pool_dim, num_init_conv_layers, kernel_size, num_all_conv_layers, d_conv_layers, encoder_dropout, d_mlp, n_mlp_layers)
    max_test_accuracy = 0

    epoch = 0
    test_loss = None
    best_test_loss = 69696969696969
    num_non_decreasing_loss = 0
    patience = 3

    while num_non_decreasing_loss < patience and epoch < NUM_EPOCH:
        train(model, train_loader, epoch)
        test_loss = validate(model, validation_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_non_decreasing_loss = 0
        else:
            num_non_decreasing_loss += 1
        epoch += 1

    return max_test_accuracy

def optimize_with_optuna():

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=objective, n_trials=20)


if __name__ == '__main__':

    main()

