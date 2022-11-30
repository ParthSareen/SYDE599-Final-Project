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
        cfg = {     'seq_length': trial.suggest_categorical(256, 512, 2048),
                    'd_model' : trial.suggest_categorical(16, 32),
                    'nhead' : trial.suggest_categorical(2,4,8),
                    'n_encoders' : trial.suggest_categorical(1,3)
                }

        train_loader, test_loader = get_loaders(cfg['train_batch_size'], cfg['test_batch_size'], cfg['aug_deg'], cfg['aug_trans'], cfg['aug_scale'])
        model = Network(cfg['activation'], trial, cfg['dropout'])

        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(func=objective, n_trials=20)



if __name__ == '__main__':
    main()

# def train_mnist(trial):

#   cfg = { 'device' : "cuda" if torch.cuda.is_available() else "cpu",
#           'train_batch_size' : trial.suggest_categorical('train_batch_size', [64, 128, 256]),
#           'test_batch_size' : 1000,
#           'aug_deg' : 20,
#           'aug_trans' : trial.suggest_uniform('aug_trans', 0, 0.3),
#           'aug_scale' : trial.suggest_uniform('aug_scale', 0, 0.3),
#           'l2_weight_decay' : trial.suggest_loguniform('l2_weight_decay', 1e-6, 1e-1),
#           'dropout' : trial.suggest_uniform('dropout', 0, 0.5),
#           'max_epochs' : 5,
#           'activation': F.relu}

#   train_loader, test_loader = get_loaders(cfg['train_batch_size'], cfg['test_batch_size'], cfg['aug_deg'], cfg['aug_trans'], cfg['aug_scale'])
#   model = Network(cfg['activation'], trial, cfg['dropout'])
#   model.to('cuda:0')
#   optimizer = optim.Adam(model.parameters(), weight_decay=cfg['l2_weight_decay'])

#   max_test_accuracy = 0
#   for epoch in range(1, cfg['max_epochs'] + 1):
#       stop = False
#       train(model, train_loader, optimizer, epoch)
#       test_accuracy = test(model, test_loader)

#       max_test_accuracy = max(max_test_accuracy,test_accuracy)
#       trial.report(max_test_accuracy, epoch)
#       if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()

#   return max_test_accuracy