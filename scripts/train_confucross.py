import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LEAKY_OR_UNFAIR_COLUMNS = {
    "package_key",
    "package_name",
    "version",
    "label",
    "malware_tgz_size",
    "snippet_count",
    "behavior_label_count",
    "evasion_label_count",
    "has_behavior_label",
    "has_evasion_label",
}

FEATURE_GROUPS = {
    "confusion_name": {
        "bare_name_len",
        "confusion_hint_count",
        "has_confusion_hint",
        "is_scoped",
        "name_digit_count",
        "name_digit_ratio",
        "name_entropy",
        "name_len",
        "name_letter_count",
        "name_separator_count",
        "name_separator_ratio",
        "name_token_count",
        "scope_len",
    },
    "metadata_dependency": {
        "dependency_count",
        "description_len",
        "dev_dependency_count",
        "has_author",
        "has_homepage",
        "has_license",
        "has_main",
        "has_package_json",
        "has_repository",
        "keyword_count",
        "optional_dependency_count",
        "peer_dependency_count",
        "script_count",
        "total_dependency_count",
    },
    "script_risk": {
        "has_install_lifecycle_script",
        "has_risky_install_script_token",
        "has_risky_script_token",
        "install_script_text_len",
        "risky_install_script_token_count",
        "risky_script_token_count",
        "script_text_len",
    },
    "source_layout": {
        "files_count",
        "js_file_count",
        "json_file_count",
        "minified_js_count",
        "package_file_count",
        "total_source_bytes",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "data/processed/eval_settings/random2000/scoped_balanced.csv",
            "data/processed/eval_settings/random2000/hard_case.csv",
        ],
    )
    parser.add_argument("--out-dir", default="runs/confucross/random2000_v1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--group-dim", type=int, default=32)
    parser.add_argument("--cross-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def feature_columns(df):
    return [
        col
        for col in df.columns
        if col not in LEAKY_OR_UNFAIR_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_group_indices(columns):
    assigned = set()
    groups = {}
    for group_name, names in FEATURE_GROUPS.items():
        idx = [i for i, col in enumerate(columns) if col in names]
        if idx:
            groups[group_name] = idx
            assigned.update(columns[i] for i in idx)
    other = [i for i, col in enumerate(columns) if col not in assigned]
    if other:
        groups["other"] = other
    return groups


def split_data(df, random_state, test_size, val_size):
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val["label"],
        random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CrossNetV2(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])

    def forward(self, x0):
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


class PlainMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class ConfuCross(nn.Module):
    def __init__(self, input_dim, group_indices, group_dim, hidden_dim, cross_layers, dropout, use_cross=True):
        super().__init__()
        self.group_indices = group_indices
        self.use_cross = use_cross
        self.group_names = list(group_indices)
        self.group_blocks = nn.ModuleDict()
        self.group_gates = nn.ModuleDict()
        for name, idx in group_indices.items():
            self.group_blocks[name] = nn.Sequential(
                nn.Linear(len(idx), group_dim),
                nn.LayerNorm(group_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.group_gates[name] = nn.Sequential(nn.Linear(len(idx), group_dim), nn.Sigmoid())

        pair_count = len(self.group_names) * (len(self.group_names) - 1) // 2
        fused_dim = group_dim * (len(self.group_names) + pair_count)
        self.cross = CrossNetV2(fused_dim, cross_layers) if use_cross else nn.Identity()
        deep_input_dim = fused_dim * 2 if use_cross else fused_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.raw_skip = nn.Linear(input_dim, 1)

    def forward(self, x):
        group_vectors = []
        for name in self.group_names:
            idx = self.group_indices[name]
            group_x = x[:, idx]
            group_vectors.append(self.group_blocks[name](group_x) * self.group_gates[name](group_x))

        pair_vectors = []
        for i in range(len(group_vectors)):
            for j in range(i + 1, len(group_vectors)):
                pair_vectors.append(group_vectors[i] * group_vectors[j])

        fused = torch.cat(group_vectors + pair_vectors, dim=1)
        if self.use_cross:
            crossed = self.cross(fused)
            deep_input = torch.cat([fused, crossed], dim=1)
        else:
            deep_input = fused
        return (self.deep(deep_input).squeeze(1) + self.raw_skip(x).squeeze(1))


def build_model(name, input_dim, group_indices, args):
    if name == "mlp_torch":
        return PlainMLP(input_dim, args.hidden_dim, args.dropout)
    if name == "confucross_no_cross":
        return ConfuCross(
            input_dim,
            group_indices,
            args.group_dim,
            args.hidden_dim,
            args.cross_layers,
            args.dropout,
            use_cross=False,
        )
    if name == "confucross":
        return ConfuCross(
            input_dim,
            group_indices,
            args.group_dim,
            args.hidden_dim,
            args.cross_layers,
            args.dropout,
            use_cross=True,
        )
    raise ValueError(f"Unknown model: {name}")


def scores_from_model(model, x, device, batch_size):
    model.eval()
    loader = make_loader(x, np.zeros(len(x)), batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def best_threshold(y_true, y_score):
    candidates = np.linspace(0.05, 0.95, 181)
    best = (0.5, -1.0)
    for threshold in candidates:
        score = f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best[0], best[1]


def evaluate(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def train_one_model(model_name, x_train, y_train, x_val, y_val, x_test, y_test, args, device):
    model = build_model(model_name, x_train.shape[1], args.group_indices, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_value = max(1.0, float((len(y_train) - y_train.sum()) / max(1, y_train.sum())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_score = scores_from_model(model, x_val, device, args.batch_size)
        threshold, val_f1 = best_threshold(y_val, val_score)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_f1": val_f1, "threshold": threshold})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    val_score = scores_from_model(model, x_val, device, args.batch_size)
    threshold, _ = best_threshold(y_val, val_score)
    test_score = scores_from_model(model, x_test, device, args.batch_size)
    return model, history, best_epoch, evaluate(y_val, val_score, threshold), evaluate(y_test, test_score, threshold), test_score


def run_setting(feature_path, args, device, out_dir):
    setting_name = Path(feature_path).stem
    setting_out = out_dir / setting_name
    setting_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_path)
    df = df[df["has_package_json"].eq(1)].copy()
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    columns = feature_columns(df)
    args.group_indices = build_group_indices(columns)

    train, val, test = split_data(df, args.random_state, args.test_size, args.val_size)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_csv(setting_out / f"{split_name}.csv", index=False)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[columns].fillna(0).astype(float)).astype(np.float32)
    x_val = scaler.transform(val[columns].fillna(0).astype(float)).astype(np.float32)
    x_test = scaler.transform(test[columns].fillna(0).astype(float)).astype(np.float32)
    y_train = train["label"].to_numpy(dtype=np.float32)
    y_val = val["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)

    all_metrics = []
    all_predictions = []
    histories = {}
    models_to_run = ["mlp_torch", "confucross_no_cross", "confucross"]
    for model_name in models_to_run:
        print(f"[{setting_name}] training {model_name} on {device}...")
        _, history, best_epoch, val_metrics, test_metrics, test_score = train_one_model(
            model_name,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            args,
            device,
        )
        histories[model_name] = history
        all_metrics.append({"setting": setting_name, "model": model_name, "split": "val", "best_epoch": best_epoch, **val_metrics})
        all_metrics.append({"setting": setting_name, "model": model_name, "split": "test", "best_epoch": best_epoch, **test_metrics})

        pred_df = test[["package_key", "package_name", "version", "label"]].copy()
        pred_df["setting"] = setting_name
        pred_df["model"] = model_name
        pred_df["score"] = test_score
        pred_df["prediction"] = (test_score >= test_metrics["threshold"]).astype(int)
        pred_df["error_type"] = np.where(
            (pred_df["label"].eq(0) & pred_df["prediction"].eq(1)),
            "false_positive",
            np.where((pred_df["label"].eq(1) & pred_df["prediction"].eq(0)), "false_negative", "correct"),
        )
        all_predictions.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(setting_out / "metrics.csv", index=False)
    predictions_df.to_csv(setting_out / "predictions.csv", index=False)
    (setting_out / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")

    return {
        "setting": setting_name,
        "feature_source": feature_path,
        "rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "feature_count": int(len(columns)),
        "features": columns,
        "feature_groups": {name: [columns[i] for i in idx] for name, idx in args.group_indices.items()},
        "metrics": metrics_df.to_dict(orient="records"),
        "outputs": {
            "metrics": str(setting_out / "metrics.csv"),
            "predictions": str(setting_out / "predictions.csv"),
            "history": str(setting_out / "training_history.json"),
        },
    }


def main():
    args = parse_args()
    set_seed(args.random_state)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for feature_path in args.features:
        summaries.append(run_setting(feature_path, args, device, out_dir))

    all_metrics = pd.concat(
        [pd.read_csv(Path(item["outputs"]["metrics"])) for item in summaries],
        ignore_index=True,
    )
    all_metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = {
        "model_family": "ConfuCross-MPD v1",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "args": {k: v for k, v in vars(args).items() if k != "group_indices"},
        "settings": summaries,
        "metrics_csv": str(out_dir / "metrics.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(all_metrics.to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
