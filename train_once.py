import torch
import torch.nn as nn
import torch.optim as optim

import Model
import FogDatasetLoader
import DataTransforms


def train(model, train_loader, optimizer, epoch):
    # device = torch.device("cuda:0")
    model.train()
    total_loss = 0
    for batch_idx, inputs, targets in zip(range(train_loader[0].shape[0]), train_loader[0], train_loader[1]):
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCELoss()(outputs, targets)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
                epoch,
                batch_idx * len(inputs),
                train_loader[0].shape[0] * len(inputs),
                loss))

    print('Training loss: {:.6f}'.format(total_loss / train_loader[0].shape[0]))


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
    # device = torch.device("cuda:0")

    model.eval()
    loss = 0
    all_tp = 0
    all_tn = 0
    all_fp = 0
    all_fn = 0
    with torch.no_grad():
        for batch_idx, inputs, targets in zip(range(test_loader[0].shape[0]), test_loader[0], test_loader[1]):
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            outputs = model(inputs)
            loss += nn.BCELoss()(outputs, targets)

            cutoff = 0.5
            predictions = torch.where(outputs > cutoff, 1, 0)
            true_positives, false_positives, true_negatives, false_negatives = confusion(predictions, targets)
            all_tp += true_positives
            all_tn += true_negatives
            all_fp += false_positives
            all_fn += false_negatives

    loss = loss / test_loader[0].shape[0]

    print('Test loss: {:.6f}; True positive: {}; True negative: {}, False Positive: {}, False negative: {}\n'.format(
        loss,
        all_tp,
        all_tn,
        all_fp,
        all_fn))

    return loss


def main():
    seq_length = 512
    d_model = 32
    nhead = 4
    d_feed_forward = 512
    n_encoders = 3
    d_input = 33
    num_conv_layers = 3
    kernel_size = 8
    encoder_dropout = 0.5
    max_pool_dim = 4
    d_mlp = 128
    n_mlp_layers = 2

    model = Model.Model(
        seq_length,
        d_model,
        nhead,
        d_feed_forward,
        n_encoders,
        d_input,
        num_conv_layers,
        kernel_size,
        encoder_dropout,
        max_pool_dim,
        d_mlp,
        n_mlp_layers,
    )
    # device = torch.device("cuda:0")
    # model.to(device)

    batch_size = 16

    fdl = FogDatasetLoader.FogDatasetLoader('./data/001')
    loader = DataTransforms.DataLoader(fdl, batch_size=1, shuffle=False)
    dt = DataTransforms.DataTransforms(loader)
    train_loader = dt.load_into_memory(batch_size)

    # for batch_idx, (inputs, targets) in enumerate(dt):
    #     print(batch_idx, inputs.shape, targets)
    # exit()

    optimizer = optim.Adam(model.parameters(), weight_decay=2e-4)

    epoch = 0
    test_loss = None
    best_test_loss = 69696969696969
    num_non_decreasing_loss = 0
    patience = 3
    while num_non_decreasing_loss < patience and epoch < 2:
        train(model, train_loader, optimizer, epoch)
        test_loss = test(model, train_loader)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_non_decreasing_loss = 0
        else:
            num_non_decreasing_loss += 1
        epoch += 1


if __name__ == '__main__':
    main()
