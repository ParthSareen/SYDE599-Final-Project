import copy

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import Model
import FogDatasetLoader
import DataTransforms


def train(model, train_loader, optimizer, epoch):
    device = torch.device("cuda:0")
    model.train()
    total_loss = 0
    for batch_idx, inputs, targets in zip(range(train_loader[0].shape[0]), train_loader[0], train_loader[1]):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCELoss()(outputs, targets)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
                epoch,
                batch_idx * len(inputs),
                train_loader[0].shape[0] * len(inputs),
                loss))

    print('Training loss: {:.6f}'.format(total_loss / train_loader[0].shape[0]))

    return total_loss / len(train_loader[0])


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def test(model, test_loader):
    device = torch.device("cuda:0")

    model.eval()
    loss = 0
    all_tp = 0
    all_tn = 0
    all_fp = 0
    all_fn = 0
    with torch.no_grad():
        for batch_idx, inputs, targets in zip(range(test_loader[0].shape[0]), test_loader[0], test_loader[1]):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss += nn.BCELoss()(outputs, targets)

            cutoff = 0.5
            predictions = torch.where(outputs >= cutoff, 1, 0)
            true_positives, false_positives, true_negatives, false_negatives = confusion(predictions, targets)
            all_tp += true_positives
            all_tn += true_negatives
            all_fp += false_positives
            all_fn += false_negatives

    loss = loss / test_loader[0].shape[0]

    accuracy = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn)
    if all_tp + all_fp != 0:
        precision = all_tp / (all_tp + all_fp)
    else:
        precision = 0
    if all_tp + all_fn != 0:
        recall = all_tp / (all_tp + all_fn)
    else:
        recall = 0

    print('Test loss: {:.6f}; True positive: {}; True negative: {}, False Positive: {}, False negative: {}, '
          'accuracy: {}, precision: {}, recall: {}\n'.format(
            loss,
            all_tp,
            all_tn,
            all_fp,
            all_fn,
            accuracy,
            precision,
            recall))

    return loss, accuracy


def main():
    seq_length = 4096
    d_model = 16
    nhead = 8
    d_feed_forward = 128
    n_encoders = 1
    d_input = 33
    num_conv_layers = 2
    max_pool_conv = 64
    kernel_size = 6
    encoder_dropout = 0.0350125571486418
    max_pool_dim = 8
    d_mlp = 16
    n_mlp_layers = 3
    dropout = 0.1676896702268

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

    batch_size = 16

    fdl = FogDatasetLoader.FogDatasetLoader('./data/training')
    loader = DataTransforms.DataLoader(fdl, batch_size=1, shuffle=False)
    dt = DataTransforms.DataTransforms(loader, window_size=seq_length, step_size=seq_length//4)
    train_loader = dt.load_into_memory(batch_size)
    train_loader = dt.normalize_data(train_loader)
    train_loader = dt.shuffle(train_loader)

    fdl = FogDatasetLoader.FogDatasetLoader('./data/validation')
    loader = DataTransforms.DataLoader(fdl, batch_size=1, shuffle=False)
    dt = DataTransforms.DataTransforms(loader, window_size=seq_length, step_size=seq_length//4)
    validation_loader = dt.load_into_memory(batch_size)
    validation_loader = dt.normalize_data(validation_loader)
    validation_loader = dt.shuffle(validation_loader)

    optimizer = optim.Adam(model.parameters(), lr=0.000292119885582925, weight_decay=4.34380331824917E-06, maximize=False)

    epoch = 0
    best_test_loss = 69696969696969
    test_losses = []
    train_losses = []
    num_non_decreasing_loss = 0
    patience = 10
    max_test_accuracy = 0
    accuracies = []
    while num_non_decreasing_loss < patience and epoch < 30:
        train_loss = train(model, train_loader, optimizer, epoch)
        train_losses.append(float(train_loss))

        test_loss, test_accuracy = test(model, validation_loader)
        test_losses.append(test_loss.cpu())
        accuracies.append(test_accuracy)

        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            torch.save(model.state_dict(), f"saved_model")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_non_decreasing_loss = 0
        else:
            num_non_decreasing_loss += 1
        epoch += 1

    model.load_state_dict(torch.load(f"saved_model"))

    plt.figure()
    plt.scatter(range(len(test_losses)), test_losses)
    plt.scatter(range(len(train_losses)), train_losses)
    plt.xlabel("Epoch number")
    plt.ylabel("loss")
    plt.legend(["test", "train"])
    plt.show()

    plt.figure()
    plt.scatter(range(len(accuracies)), [x*100 for x in accuracies])
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy [%]")
    plt.show()

    preds = []
    truths = []
    for inputs, targets in zip(validation_loader[0], validation_loader[1]):
        inputs = inputs.to(device)
        preds.append([pred[0] for pred in model(inputs).tolist()])
        truths.append([truth[0] for truth in targets.tolist()])

    preds = np.array(preds).reshape(-1)
    truths = np.array(truths).reshape(-1)

    print("preds", preds)
    print("truths", truths)

    plt.figure()
    plt.scatter(range(len(preds)), preds, s=5)
    plt.scatter(range(len(truths)), truths, s=5)
    plt.legend(["preds", "targets"])
    plt.show()


if __name__ == '__main__':
    main()
