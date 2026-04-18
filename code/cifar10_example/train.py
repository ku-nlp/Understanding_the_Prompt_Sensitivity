import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json

import tqdm

seed = 1024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 10             
epochs = 20                 
alpha_ce = 1.0               
alpha_triplet = 0.2          
best_metric_name = "macro_f1"

transform = transforms.Compose([
    transforms.Resize(112), transforms.CenterCrop(112),
    transforms.ToTensor()
])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4, drop_last=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=4)

backbone = models.resnet101(weights=None)
backbone.fc = nn.Identity()
backbone = backbone.to(device)

LAYER1_BLOCK_NAMES = [f'layer1_block{i}' for i in range(len(backbone.layer1))]
LAYER2_BLOCK_NAMES = [f'layer2_block{i}' for i in range(len(backbone.layer2))]
LAYER3_BLOCK_NAMES = [f'layer3_block{i}' for i in range(len(backbone.layer3))]
LAYER4_BLOCK_NAMES = [f'layer4_block{i}' for i in range(len(backbone.layer4))]
FEAT_NAMES = (
    ['input']
    + LAYER1_BLOCK_NAMES + ['layer1']
    + LAYER2_BLOCK_NAMES + ['layer2']
    + LAYER3_BLOCK_NAMES + ['layer3']
    + LAYER4_BLOCK_NAMES + ['layer4']
    + ['proj', 'fc']
)

proj = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.PReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, 128)
).to(device)

clf = nn.Linear(128, num_classes).to(device)

params = list(backbone.parameters()) + list(proj.parameters()) + list(clf.parameters())
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

def batch_all_triplets(emb, labels, margin=0.2):
    # emb: [B,D], labels: [B]
    dist = torch.cdist(emb, emb)  # [B,B]
    lbl_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_mask = lbl_eq & (~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device))
    neg_mask = ~lbl_eq

    ap_dist = dist.unsqueeze(2)  # [B,B,1]
    an_dist = dist.unsqueeze(1)  # [B,1,B]
    triplet_loss = ap_dist - an_dist + margin  # [B,B,B]
    mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
    triplet_loss = torch.clamp(triplet_loss[mask], min=0.0)
    if triplet_loss.numel() == 0:
        return emb.new_tensor(0.)
    return triplet_loss.mean()

layer_feats = {}
def reg_hook(name):
    def _hook(m, x, y):
        # y: [B, C, H, W] -> GAP to [B, C]
        feat = F.adaptive_avg_pool2d(y, 1).squeeze(-1).squeeze(-1)
        layer_feats[name] = feat
    return _hook

backbone.layer1.register_forward_hook(reg_hook('layer1'))
for i, block in enumerate(backbone.layer1):
    block.register_forward_hook(reg_hook(f'layer1_block{i}'))
backbone.layer2.register_forward_hook(reg_hook('layer2'))
for i, block in enumerate(backbone.layer2):
    block.register_forward_hook(reg_hook(f'layer2_block{i}'))
backbone.layer3.register_forward_hook(reg_hook('layer3'))
for i, block in enumerate(backbone.layer3):
    block.register_forward_hook(reg_hook(f'layer3_block{i}'))
backbone.layer4.register_forward_hook(reg_hook('layer4'))
for i, block in enumerate(backbone.layer4):
    block.register_forward_hook(reg_hook(f'layer4_block{i}'))

@torch.no_grad()
def forward_collect(imgs):
   
    B = imgs.size(0)
    x_input = imgs.view(B, -1)              
    x_input = F.normalize(x_input, dim=-1)  

    layer_feats.clear()
    _ = backbone(imgs)                      

    feats_l4 = layer_feats['layer4']        
    emb = F.normalize(proj(feats_l4), dim=-1)  
    logits = clf(emb)                         

    feats_dict = {'input': x_input}
    for name in LAYER1_BLOCK_NAMES:
        feats_dict[name] = F.normalize(layer_feats[name], dim=-1)
    feats_dict['layer1'] = F.normalize(layer_feats['layer1'], dim=-1)
    for name in LAYER2_BLOCK_NAMES:
        feats_dict[name] = F.normalize(layer_feats[name], dim=-1)
    feats_dict['layer2'] = F.normalize(layer_feats['layer2'], dim=-1)
    for name in LAYER3_BLOCK_NAMES:
        feats_dict[name] = F.normalize(layer_feats[name], dim=-1)
    feats_dict['layer3'] = F.normalize(layer_feats['layer3'], dim=-1)
    for name in LAYER4_BLOCK_NAMES:
        feats_dict[name] = F.normalize(layer_feats[name], dim=-1)
    feats_dict['layer4'] = F.normalize(layer_feats['layer4'], dim=-1)
    feats_dict['proj'] = emb
    feats_dict['fc'] = F.normalize(logits, dim=-1)
    return feats_dict, emb, logits

def train_epoch_supervised(alpha_ce=1.0, alpha_triplet=0.0):
    backbone.train(); proj.train(); clf.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()

        layer_feats.clear()
        _ = backbone(imgs)
        feats = layer_feats['layer4']                  
        emb = F.normalize(proj(feats), dim=-1)         

        logits = clf(emb)                              
        loss_ce = F.cross_entropy(logits, labels)

        if alpha_triplet > 0:
            loss_tri = batch_all_triplets(emb, labels, margin=0.2)
        else:
            loss_tri = emb.new_tensor(0.0)

        loss = alpha_ce * loss_ce + alpha_triplet * loss_tri
        loss.backward()
        opt.step()

@torch.no_grad()
def eval_all_layers_intra_inter_by_label(max_batches=512):
    
    backbone.eval(); proj.eval(); clf.eval()
    names = FEAT_NAMES

    def _empty_per_label():
        return {str(i): {k: {'mean': 0.0, 'std': 0.0} for k in names} for i in range(num_classes)}

    state = {
        'intra': _empty_per_label(),
        'inter': _empty_per_label()
    }

    counts = 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        feats_dict, _, _ = forward_collect(imgs)

        B = labels.size(0)
        if B <= 1:
            counts += 1
            if counts >= max_batches:
                break
            continue

        uniq = labels.unique()
        for name in names:
            x = feats_dict[name]             
            dist_full = torch.cdist(x, x)   

            for y in uniq:
                y_int = int(y.item())
                idx_pos = (labels == y).nonzero(as_tuple=True)[0]
                idx_neg = (labels != y).nonzero(as_tuple=True)[0]

                if idx_pos.numel() > 1:
                    d_pp = dist_full.index_select(0, idx_pos).index_select(1, idx_pos)
                    n = d_pp.size(0)
                    uptri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=d_pp.device), diagonal=1)
                    vals_intra = d_pp[uptri]
                    if vals_intra.numel() > 0:
                        state['intra'][str(y_int)][name]['mean'] = torch.mean(vals_intra).detach().cpu().item()
                        state['intra'][str(y_int)][name]['std'] = torch.std(vals_intra).detach().cpu().item()

                if idx_pos.numel() > 0 and idx_neg.numel() > 0:
                    d_pn = dist_full.index_select(0, idx_pos).index_select(1, idx_neg)  # [|pos|, |neg|]
                    vals_inter = d_pn.reshape(-1)
                    if vals_inter.numel() > 0:
                        state['inter'][str(y_int)][name]['mean'] = torch.mean(vals_inter).detach().cpu().item()
                        state['inter'][str(y_int)][name]['std'] = torch.std(vals_inter).detach().cpu().item()

        counts += 1
        if counts >= max_batches:
            break

    return state

@torch.no_grad()
def eval_supervised_metrics():
    backbone.eval(); proj.eval(); clf.eval()

    tp = torch.zeros(num_classes, dtype=torch.long).to(device)
    fp = torch.zeros(num_classes, dtype=torch.long).to(device)
    fn = torch.zeros(num_classes, dtype=torch.long).to(device)
    support = torch.zeros(num_classes, dtype=torch.long).to(device)
    total_correct = 0
    total = 0

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        layer_feats.clear()
        _ = backbone(imgs)
        feats = layer_feats['layer4']
        emb = F.normalize(proj(feats), dim=-1)
        logits = clf(emb)
        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total += labels.numel()

        for c in range(num_classes):
            c_true = (labels == c)
            c_pred = (preds == c)
            tp[c] += (c_true & c_pred).sum()
            fp[c] += (~c_true & c_pred).sum()
            fn[c] += (c_true & ~c_pred).sum()
            support[c] += c_true.sum()

    eps = 1e-12
    precision = tp.float() / (tp + fp + eps).float()
    recall    = tp.float() / (tp + fn + eps).float()
    f1        = 2 * precision * recall / (precision + recall + eps)

    per_class = {
        str(c): {
            'support': int(support[c].item()),
            'precision': float(precision[c].item()),
            'recall': float(recall[c].item()),
            'f1': float(f1[c].item()),
        } for c in range(num_classes)
    }

    overall_acc = total_correct / max(total, 1)
    macro_f1 = float(f1.mean().item())

    result = {
        'overall_acc': overall_acc,
        'macro_f1': macro_f1,
        'per_class': per_class
    }
    return result

@torch.no_grad()
def eval_supervised_metrics_per_label():
    
    backbone.eval(); proj.eval(); clf.eval()

    all_preds = []
    all_labels = []
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        layer_feats.clear()
        _ = backbone(imgs)
        feats = layer_feats['layer4']
        emb = F.normalize(proj(feats), dim=-1)
        logits = clf(emb)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds)   
    y_true = torch.cat(all_labels)  
    N = y_true.numel()

    overall_acc = (y_pred == y_true).float().mean().item()

    per_class = {}
    f1s = []
    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)

        tp = (true_c & pred_c).sum().item()
        fp = (~true_c & pred_c).sum().item()
        fn = (true_c & ~pred_c).sum().item()
        tn = N - tp - fp - fn
        support = int(true_c.sum().item())

        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)

        acc_in_class = recall
        spec    = tn / (tn + fp + eps)
        ovr_acc = (tp + tn) / (tp + tn + fp + fn + eps)

        per_class[str(c)] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "acc_in_class": float(acc_in_class),
            "spec": float(spec),
            "ovr_acc": float(ovr_acc),
        }
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / max(len(f1s), 1))

    return {
        "overall_acc": float(overall_acc),
        "macro_f1": macro_f1,
        "per_class": per_class
    }

def main():
    best_metric = -1.0
    best_epoch = -1
    save_results = []
    for ep in tqdm.tqdm(range(epochs), total=epochs):
        train_epoch_supervised(alpha_ce=alpha_ce, alpha_triplet=alpha_triplet)

        state = eval_all_layers_intra_inter_by_label(max_batches=512)

        result = eval_supervised_metrics()
        result_per_label = eval_supervised_metrics_per_label()

        current_metric = result[best_metric_name]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = ep

        distance = {
            'intra': {
                str(c): {name: state['intra'][str(c)][name] for name in FEAT_NAMES}
                for c in range(num_classes)
            },
            'inter': {
                str(c): {name: state['inter'][str(c)][name] for name in FEAT_NAMES}
                for c in range(num_classes)
            }
        }
        mean_f1 = result['macro_f1']

        save_results.append(
            {
                "epoch": ep+1,
                "distance": distance,
                "mean_f1": mean_f1,
                "mean_acc": result['overall_acc']
            }
        )
        print(f"Epoch {ep+1}  Acc={result['overall_acc']:.4f}  Macro-F1={result['macro_f1']:.4f}")

    print(f"==> Best Epoch: {best_epoch}  Best {best_metric_name}: {best_metric:.4f}")
    
    out_dir = "results/data_results/cifar10_all_layers"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cifar10_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in save_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
