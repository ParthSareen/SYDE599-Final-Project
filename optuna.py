import DataTransforms
import FogDatasetLoader
import torch
import Model
from torch.utils.data import DataLoader
import optuna
import torch.optim as optim
from optuna.trial import TrialState
#import train
#import validate

def main():
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=objective, n_trials=20)
    optuna_trial(study) #not sure if this is correct

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
    d_input = 33
    kernel_size = 8
    num_conv_layers = 3
    encoder_dropout = 0.5
    max_pool_dim = 4
    d_mlp = 128
    n_mlp_layers = 2

    model = Model(
        cfg['seq_length'],
        cfg['d_model'],
        cfg['nhead'],
        d_feed_forward,
        cfg['n_encoders'],
        d_input,
        num_conv_layers,
        kernel_size,
        encoder_dropout,
        max_pool_dim,
        d_mlp,
        n_mlp_layers,)

    optimizer = optim.Adam(model.parameters(), weight_decay=2e-4)
    batch_size=1

    #should this be somewhere else?
    fdl = FogDatasetLoader.FogDatasetLoader('./data')
    loader = DataLoader(fdl, batch_size, shuffle=False)
    dt = DataTransforms(loader)
    train_loader = dt.load_into_memory() #TODO change
    validation_loader = dt.load_into_memory() #TODO change

    epoch = 0
    test_loss = None
    best_test_loss = 69696969696969
    num_non_decreasing_loss = 0
    patience = 3
    max_test_accuracy = 0

    while num_non_decreasing_loss < patience and epoch < NUM_EPOCH:
        train(model, train_loader, optimizer, epoch)
        test_loss = validate(model, validation_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_non_decreasing_loss = 0
        else:
            num_non_decreasing_loss += 1
        epoch += 1

    return max_test_accuracy



def optuna_trial(study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# if __name__ == '__main__':

#     main()

