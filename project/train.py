import argparse
import os
import time
import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from model.model import VGAEModel
from data.input_data import load_data
from model.preprocess import (
    mask_test_edges,
    preprocess_graph,
    sparse_to_tuple,
)
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
from torch.utils.tensorboard import SummaryWriter


warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--learning_rate", type=float, default=0.01, help="Initial learning rate."
)
parser.add_argument(
    "--epochs", "-e", type=int, default=350, help="Number of epochs to train."
)
parser.add_argument(
    "--hidden1",
    "-h1",
    type=int,
    default=32,
    help="Number of units in hidden layer 1.",
)
parser.add_argument(
    "--hidden2",
    "-h2",
    type=int,
    default=16,
    help="Number of units in hidden layer 2.",
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")
parser.add_argument("--cpu_only", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--hp_tuning", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()


# check device
if (args.cpu_only):
    device = torch.device("cpu")
else:
    device = torch.device(
        "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
    )


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    rec_acc = 0.0
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]].item())
        if adj_rec[e[0], e[1]] > 0.5:
            rec_acc += 1
    rec_acc /= len(edges_pos)

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]].item())

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score, rec_acc


def train(adj, features):
    features = sparse_to_tuple(features.tocoo())

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix(
        (adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape
    )
    adj_orig.eliminate_zeros()

    (
        adj_train,
        train_edges,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_normalization, adj_norm = preprocess_graph(adj)

    # Create model
    graph = dgl.from_scipy(adj_normalization).to(device)
    graph.add_self_loop()

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_label = torch.sparse.FloatTensor(
        torch.LongTensor(adj_label[0].T),
        torch.FloatTensor(adj_label[1]),
        torch.Size(adj_label[2]),
    ).to(device)
    features = torch.sparse.FloatTensor(
        torch.LongTensor(features[0].T),
        torch.FloatTensor(features[1]),
        torch.Size(features[2]),
    ).to(device)

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    features = features.to_dense()
    in_dim = features.shape[-1]

    vgae_model = VGAEModel(in_dim, args.hidden1, args.hidden2, device)
    vgae_model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print(
        "Total Parameters:",
        sum([p.nelement() for p in vgae_model.parameters()]),
    )

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits = vgae_model.forward(graph, features)

        # compute loss
        loss = norm * F.binary_cross_entropy(
            logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor
        )
        kl_divergence = (0.5 / logits.size(0) * (1 + 2 * vgae_model.log_std - vgae_model.mean**2 - torch.exp(vgae_model.log_std) ** 2)
                         .sum(1).mean())
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj_label)

        val_roc, val_ap, val_rec_acc = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(loss.item()),
            "train_acc=",
            "{:.5f}".format(train_acc),
            "val_roc=",
            "{:.5f}".format(val_roc),
            "val_ap=",
            "{:.5f}".format(val_ap),
            "val_rec_acc=",
            "{:.5f}".format(val_rec_acc),
            "time=",
            "{:.5f}".format(time.time() - t),
        )

        writer.add_scalar('Training loss', loss, epoch)
        writer.add_scalar('Training accuracy', train_acc, epoch)
        writer.add_scalar('ROC Score (validation)', val_roc, epoch)
        writer.add_scalar('Average Precision Score (validation)', val_ap, epoch)
        writer.add_scalar('Reconstruction Accuracy (validation)', val_rec_acc, epoch)

    test_roc, test_ap, test_rec_acc = get_scores(test_edges, test_edges_false, logits)
    print(
        "End of training!",
        "test_rec_acc=",
        "{:.5f}".format(test_rec_acc),
        "test_roc=",
        "{:.5f}".format(test_roc),
        "test_ap=",
        "{:.5f}".format(test_ap),
    )

    if args.hp_tuning:
        writer.add_hparams({'Epochs': args.epochs,
                            'Learning rate': args.learning_rate,
                            'N of units in hidden1 layer': args.hidden1,
                            'N of units in hidden2 layer': args.hidden2},
                            {'Loss': loss,
                             'ROC Score': test_rec_acc,
                             'Average Precision Score': test_ap,
                             'Reconstruction Accuracy': test_rec_acc})
    else:
        writer.add_scalar('ROC Score (test)', test_rec_acc)
        writer.add_scalar('Average Precision Score (test)', test_ap)
        writer.add_scalar('Reconstruction Accuracy (test)', test_rec_acc)

def hyperparam_tuning(adj, features):
    for epochs in [100, 200, 300]:
        for lr in [0.001, 0.01, 0.1]:
            for h1 in [16, 32, 64]:
                for h2 in [16, 32, 64]:
                    args.epochs = epochs
                    args.learning_rate = lr
                    args.hidden1 = h1
                    args.hidden2 = h2
                    train(adj, features)

if __name__ == "__main__":
    writer = SummaryWriter()
    adj, features = load_data()
    if args.hp_tuning:
        hyperparam_tuning(adj, features)
    else:
        train(adj, features)
    writer.flush()
    writer.close()
