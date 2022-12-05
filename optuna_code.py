import DataTransforms
import FogDatasetLoader
import torch
import Model
from torch.utils.data import DataLoader
import optuna
import torch.optim as optim
from optuna.trial import TrialState
from train_once import train, test
import sklearn


seq_length = 4096
batch_size = 16

fdl = FogDatasetLoader.FogDatasetLoader('./data/training')
loader = DataTransforms.DataLoader(fdl, batch_size=1, shuffle=False)
dt = DataTransforms.DataTransforms(loader, window_size=seq_length, step_size=seq_length // 4)
train_loader = dt.load_into_memory(batch_size)
train_loader = dt.normalize_data(train_loader)
train_loader = dt.shuffle(train_loader)

fdl = FogDatasetLoader.FogDatasetLoader('./data/validation')
loader = DataTransforms.DataLoader(fdl, batch_size=1, shuffle=False)
dt = DataTransforms.DataTransforms(loader, window_size=seq_length, step_size=seq_length // 4)
validation_loader = dt.load_into_memory(batch_size)
validation_loader = dt.normalize_data(validation_loader)
validation_loader = dt.shuffle(validation_loader)


def main():
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=objective, n_trials=300)
    return optuna_trial(study)  # not sure if this is correct


def objective(trial):
    print(f"starting trial {trial._trial_id}")
    # optuna: window size, sequence length, d_model, nhead,n_encoders
    # constant parameters: d_feed_forward: kernel size, d_conv_layers, step size = 1/4 window
    cfg = {'d_model': trial.suggest_categorical('d_model', [16, 32, 64]),
           'nhead': trial.suggest_categorical('nhead', [2, 4, 8]),
           'n_encoders': trial.suggest_int('n_encoders', 1, 3),
           'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
           'weight_decay': trial.suggest_loguniform("weight_decay", 1e-6, 1e-4),
           "dropout": trial.suggest_float("dropout", 0, 0.5),
           "max_pool_conv": trial.suggest_categorical("max_pool_conv", [16, 32, 64, 128]),
           "kernel_size": trial.suggest_int("kernel_size", 4, 32),
           "d_mlp": trial.suggest_categorical("d_mlp", [16, 32, 62, 128]),
           "num_conv_layers": trial.suggest_int("num_conv_layers", 2, 8),
           "encoder_dropout": trial.suggest_float("encoder_dropout", 0, 0.5),
           "d_feed_forward": trial.suggest_categorical("d_feed_forward", [64, 128, 256, 512]),
           "max_pool_dim": trial.suggest_categorical("max_pool_dim", [4, 8, 16]),
           "n_mlp_layers": trial.suggest_int("n_mlp_layers", 2, 4)
           }

    d_model = cfg["d_model"]
    nhead = cfg["nhead"]
    d_feed_forward = cfg["d_feed_forward"]
    n_encoders = cfg["n_encoders"]
    d_input = 33
    num_conv_layers = cfg["num_conv_layers"]
    max_pool_conv = cfg["max_pool_conv"]
    kernel_size = cfg["kernel_size"]
    encoder_dropout = cfg["encoder_dropout"]
    max_pool_dim = cfg["max_pool_dim"]
    d_mlp = cfg["d_mlp"]
    n_mlp_layers = cfg["n_mlp_layers"]
    dropout = cfg["dropout"]

    model = Model.Model(
        seq_length,
        d_model,
        nhead,
        d_feed_forward,
        n_encoders,
        d_input,
        num_conv_layers,
        max_pool_conv,
        kernel_size,
        encoder_dropout,
        max_pool_dim,
        d_mlp,
        n_mlp_layers,
        dropout,
    )
    device = torch.device("cuda:0")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=cfg["weight_decay"], lr=cfg["learning_rate"])

    epoch = 0
    test_loss = None
    best_test_loss = 69696969696969
    num_non_decreasing_loss = 0
    patience = 10
    max_test_accuracy = 0

    while num_non_decreasing_loss < patience and epoch < 30:
        train(model, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, validation_loader)

        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            torch.save(model.state_dict(), f"saved_models/trial_{trial._trial_id}_best")

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

    df = study.trials_dataframe().sort_values('value')
    df.to_csv('saved_models/optuna_results_1.csv', index=False)

    f = open("saved_models/importances.txt", "w")
    for key, val in optuna.importance.get_param_importances(study, target=None).items():
        print(f'{key}, {val * 100}\n')
        f.write(f"{key}, {val}")
    f.close()


    return trial.params.items()


if __name__ == '__main__':
    main()
