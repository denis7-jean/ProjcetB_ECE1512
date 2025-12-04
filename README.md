# **ECE1512 — Project B: Multi-Instance Learning (MIL) on BRACS**

**University of Toronto — ECE1512 (Machine Learning in Computer Vision)**
**Author: Dennis Lan (denis7-jean)**
This repository contains the full implementation, experiments, and results for **Project B — Multi-Instance Learning**, including:

* Baseline training using ABMIL
* Two ablation studies (architecture & loss function)
* Reproducible configs, code, results, and plots

---

# **1. Project Structure Overview**

```
ProjectB_ECE1512/
│
├── main.py                     # Training script (modified)
├── model.py                    # ABMIL model (with pooling modifications)
├── dataloader.py               # HDF5 MIL dataset loader
├── utils.py                    # Accuracy, metric logger, utilities
│
├── config/                     # YAML configs for BRACS / Camelyon
│   ├── bracs_medical_ssl_config.yml
│   ├── camelyon16_medical_ssl_config.yml
│   └── camelyon17_medical_ssl_config.yml
│
├── results/                    # Baseline & Ablation outputs
│   ├── baseline_bracs_best.json
│   ├── baseline_bracs_history.json
│   ├── baseline_bracs_val_acc.png
│   ├── baseline_bracs_val_auc.png
│   ├── baseline_bracs_val_f1.png
│   ├── baseline_bracs_val_loss.png
│   ├── ablation_bracs_max_ce_best.json
│   ├── ablation_bracs_max_ce_history.json
│   ├── ablation_bracs_attention_focal_best.json
│   └── ablation_bracs_attention_focal_history.json
│
└── dataset_csv/                # BRACS slide-to-label mapping
    └── bracs.csv
```

---

# **2. Baseline: Attention-based MIL (ABMIL)**

### **Model components**

| Component      | Role                                                                  |
| -------------- | --------------------------------------------------------------------- |
| **Encoder**    | Pre-trained ViT-S/16 (Medical SSL), extracts 384-dim patch embeddings |
| **Aggregator** | Attention MIL pooling (Ilse et al., 2018)                             |
| **Classifier** | 2-layer MLP → 3-class softmax                                         |

### **Training details**

* Dataset: **BRACS**
* Optimizer: **AdamW**
* Epochs: **50**
* LR schedule: warmup + cosine decay
* Loss: **Cross-Entropy**
* Batch size: 1 bag per iteration (standard MIL)

### **Baseline Results (Validation)**

| Metric   | Best                    |
| -------- | ----------------------- |
| Accuracy | **42.2%**               |
| AUC      | **0.541**               |
| F1       | **0.36**                |
| Loss     | decreasing to **1.075** |

Plots are included in `/results/*.png`:

* Validation Accuracy
* Validation AUC
* Macro F1
* Validation Loss

---

# **3. Ablation Studies**

## **A) Architecture Ablation — Replace Attention with Max Pooling**

Modified in `model.py`:

```python
if conf.pooling_mode == "max":
    bag_feat, _ = torch.max(feats, dim=1)
else:
    bag_feat = attention_pooling(...)
```

### **Result Summary**

| Model                           | Val Acc | Val AUC | Val F1 |
| ------------------------------- | ------- | ------- | ------ |
| **Baseline: ABMIL (attention)** | 42.2%   | 0.541   | 0.36   |
| **Max Pooling**                 | 30.6%   | 0.417   | 0.23   |

**Observation:**
Max pooling performs significantly worse → confirms **attention is essential** for learning discriminative patch importance in histopathology MIL.

---

## **B) Loss Function Ablation — Focal Loss**

Modified in `main.py`:

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        ...
```

### **Result Summary**

| Model           | Val Acc | Val AUC | Val F1 |
| --------------- | ------- | ------- | ------ |
| **Baseline CE** | 42.2%   | 0.541   | 0.36   |
| **Focal Loss**  | 39.4%   | 0.518   | 0.34   |

**Observation:**
Focal loss slightly decreases performance.
This matches intuition → BRACS is *not* extremely imbalanced, so CE remains more stable.

---

# **4. Quantitative Summary Table**

### **All Experiments Comparison**

| Experiment            | Pooling   | Loss  | AUC       | Acc       | F1       |
| --------------------- | --------- | ----- | --------- | --------- | -------- |
| **Baseline**          | Attention | CE    | **0.541** | **42.2%** | **0.36** |
| Architecture Ablation | Max       | CE    | 0.417     | 30.6%     | 0.23     |
| Loss Ablation         | Attention | Focal | 0.518     | 39.4%     | 0.34     |

---

# **5. How to Reproduce**

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run baseline

```bash
python main.py --config config/bracs_medical_ssl_config.yml
```

### 3. Architecture Ablation (Max Pool)

```bash
python main.py --config config/bracs_medical_ssl_config.yml \
    --pooling_mode max \
    --loss_type ce
```

### 4. Loss Ablation (Focal)

```bash
python main.py --config config/bracs_medical_ssl_config.yml \
    --pooling_mode attention \
    --loss_type focal
```

---

# **6. Notes on Code Modifications**

To support ablations, we modified:

### ✔ main.py

* Added `--pooling_mode` and `--loss_type` arguments
* Added FocalLoss implementation
* Replaced torchmetrics with sklearn metrics for compatibility

### ✔ model.py

* Added `max pooling` branch
* Clean separation of attention pooling vs max pooling

### ✔ results/

* All history & best metrics saved per experiment
* All baseline plots generated from history json

---

# **7. References**

* Ilse et al., *Attention-based Deep Multiple Instance Learning* (ICML 2018)
* ACMIL implementation inspiration: [https://github.com/dazhangyu123/ACMIL](https://github.com/dazhangyu123/ACMIL)
* BRACS dataset paper
