import json
import os
import pprint
from dataloader import build_HDF5_feat_dataset
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from utils import Struct, MetricLogger, accuracy
from model import ABMIL
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, criterion, data_loader, optimizer,
                    device):
    model.train()
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32)
        batch_size = bag.shape[0]
        label = data['label'].to(device)
        train_logits = model(bag)
        train_loss = criterion(train_logits.view(batch_size, -1), label)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf):
    model.eval()
    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter="  ")

    for data in data_loader:
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)

        slide_preds = model(image_patches)
        loss = criterion(slide_preds, labels)

        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, labels, topk=(1,))[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])

        y_pred.append(pred)
        y_true.append(labels)

    # combine to a big tensor
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    #convert to numpy，easy for sklearn computation
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    # ===== AUC (multiclass) =====
    try:
        auroc = roc_auc_score(y_true_np, y_pred_np, multi_class='ovo')
    except Exception:
        auroc = 0.0

    # ===== F1 (macro) =====
    pred_labels = y_pred_np.argmax(axis=1)
    f1 = f1_score(y_true_np, pred_labels, average='macro')

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1,
                  losses=metric_logger.loss,
                  AUROC=auroc,
                  F1=f1))

    return auroc, metric_logger.acc1.global_avg, f1, metric_logger.loss.global_avg




def main(args):
    
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        print("Used config:", c, flush=True)
        conf = Struct(**c)
    
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf=conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    
    model = ABMIL(conf=conf, D=conf.D_feat)

    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=conf.lr, 
                                  weight_decay=conf.wd)
    
    sched_warmup = LinearLR(
        optimizer,
        start_factor=0.01,  # ~0.0 to avoid zero-LR edge cases
        end_factor=1.0,
        total_iters=conf.warmup_epoch
    )

    # Cosine from base_lr -> eta_min
    sched_cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, conf.train_epoch - conf.warmup_epoch),
        eta_min=1e-10
    )

    # Call warmup for first `warmup_epochs`, then cosine after
    scheduler = SequentialLR(
        optimizer,
        schedulers=[sched_warmup, sched_cosine],
        milestones=[conf.warmup_epoch]
    )

    # Display the configuration settings
    print('Configuration:')
    pprint.pprint(conf)
    if conf.loss_type == "focal":
        print("Using Focal Loss")
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0):
                super().__init__()
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(reduction='none')

            def forward(self, inputs, targets):
                ce = self.ce(inputs, targets)
                pt = torch.exp(-ce)
                loss = (1-pt)**self.gamma * ce
                return loss.mean()
        criterion = FocalLoss(gamma=2.0)

    else:
        print("Using Cross Entropy")
        criterion = nn.CrossEntropyLoss()

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
        # ====== History dict for logging per-epoch metrics ======
    history = {
        "epoch": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": [],
        "val_loss": []
    }

    for epoch in range(conf.train_epoch):
        train_one_epoch(model, criterion, train_loader, optimizer,
                        device)
        
        val_auc, val_acc, val_f1, val_loss = evaluate(model, criterion, val_loader, device, conf)
        test_auc, test_acc, test_f1, test_loss = evaluate(model, criterion, test_loader, device, conf)
        scheduler.step()
        # ====== Log into history ======
        history["epoch"].append(epoch + 1)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["val_loss"].append(val_loss)
        # ===============================
        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1

    print("Results on best epoch:")
    print(best_state)

    # ====== Save results & history ======
    os.makedirs("results", exist_ok=True)

    # if Baseline (attention + ce), use former naming
    if conf.pooling_mode == "attention" and conf.loss_type == "ce":
        best_path = "results/baseline_bracs_best.json"
        hist_path = "results/baseline_bracs_history.json"
    else:
        # Others reckon as ablation
        exp_tag = f"{conf.dataset.lower()}_{conf.pooling_mode}_{conf.loss_type}"
        best_path = f"results/ablation_{exp_tag}_best.json"
        hist_path = f"results/ablation_{exp_tag}_history.json"

    with open(best_path, "w") as f:
        json.dump(best_state, f, indent=2)

    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best results to {best_path}")
    print(f"Saved history to {hist_path}")
    # ====================================



def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument(
        '--config',
        dest='config',
        default='config/camelyon_config.yml',
        help='settings of Tip-Adapter in yaml format'
    )

    # NEW: pooling mode for architecture ablation
    parser.add_argument(
        '--pooling_mode',
        default='attention',
        choices=['attention', 'max'],
        help='MIL pooling operator: attention (baseline) or max (ablation)'
    )

    # NEW: loss type for loss ablation
    parser.add_argument(
        '--loss_type',
        default='ce',
        choices=['ce', 'focal'],
        help='Loss function: ce (CrossEntropy) or focal'
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    main(args)