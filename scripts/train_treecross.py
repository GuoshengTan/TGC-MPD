import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
    parser.add_argument("--out-dir", default="runs/treecross/random2000_v1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--leaf-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rf-trees", type=int, default=160)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--model-family", default="TreeCross-MPD")
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


def choose_blend(y_val, left_score, right_score, step):
    best = {"alpha": 0.0, "threshold": 0.5, "f1": -1.0}
    for alpha in np.arange(0.0, 1.0 + 1e-9, step):
        score = alpha * left_score + (1.0 - alpha) * right_score
        threshold, val_f1 = best_threshold(y_val, score)
        if val_f1 > best["f1"]:
            best = {"alpha": float(alpha), "threshold": float(threshold), "f1": float(val_f1)}
    return best


def choose_confidence_gate(y_val, teacher_score, fallback_score, step):
    best = {"confidence_margin": 0.0, "threshold": 0.5, "f1": -1.0}
    teacher_confidence = np.abs(teacher_score - 0.5)
    for margin in np.arange(0.0, 0.501, step):
        use_teacher = teacher_confidence >= margin
        score = np.where(use_teacher, teacher_score, fallback_score)
        threshold, val_f1 = best_threshold(y_val, score)
        if val_f1 > best["f1"]:
            best = {"confidence_margin": float(margin), "threshold": float(threshold), "f1": float(val_f1)}
    return best


def apply_confidence_gate(teacher_score, fallback_score, margin):
    use_teacher = np.abs(teacher_score - 0.5) >= margin
    return np.where(use_teacher, teacher_score, fallback_score)


def fit_tree_teachers(x_train_raw, y_train, random_state, rf_trees):
    rf = RandomForestClassifier(
        n_estimators=rf_trees,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=random_state,
    )
    rf.fit(x_train_raw, y_train)
    hgb.fit(x_train_raw, y_train)
    return rf, hgb


def teacher_aux_features(rf, hgb, x_raw):
    rf_prob = rf.predict_proba(x_raw)[:, 1]
    hgb_prob = hgb.predict_proba(x_raw)[:, 1]
    avg_prob = (rf_prob + hgb_prob) / 2.0
    disagreement = np.abs(rf_prob - hgb_prob)
    return np.column_stack([rf_prob, hgb_prob, avg_prob, disagreement]).astype(np.float32)


def build_leaf_maps(train_leaves):
    maps = []
    offsets = []
    current = 0
    for col in range(train_leaves.shape[1]):
        unique = sorted(np.unique(train_leaves[:, col]).tolist())
        mapping = {leaf_id: current + i for i, leaf_id in enumerate(unique)}
        maps.append(mapping)
        offsets.append(current)
        current += len(unique)
    return maps, current


def map_leaves(leaves, maps):
    mapped = np.zeros_like(leaves, dtype=np.int64)
    for col, mapping in enumerate(maps):
        mapped[:, col] = [mapping[int(v)] for v in leaves[:, col]]
    return mapped


class TreeTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dense_x, aux_x, leaf_ids, y):
        self.dense_x = torch.tensor(dense_x, dtype=torch.float32)
        self.aux_x = torch.tensor(aux_x, dtype=torch.float32)
        self.leaf_ids = torch.tensor(leaf_ids, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.dense_x[idx], self.aux_x[idx], self.leaf_ids[idx], self.y[idx]


def make_loader(dense_x, aux_x, leaf_ids, y, batch_size, shuffle):
    return DataLoader(TreeTensorDataset(dense_x, aux_x, leaf_ids, y), batch_size=batch_size, shuffle=shuffle)


class TreeCrossMPD(nn.Module):
    def __init__(self, dense_dim, aux_dim, total_leaves, n_trees, leaf_dim, hidden_dim, dropout, mode):
        super().__init__()
        self.mode = mode
        self.n_trees = n_trees
        self.use_leaf = mode in {"treecross_leaf", "treecross_full"}
        self.use_aux = mode in {"treecross_prior", "treecross_full"}

        self.dense_encoder = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.use_aux:
            self.aux_encoder = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU(), nn.Dropout(dropout))
        if self.use_leaf:
            self.leaf_embedding = nn.Embedding(total_leaves, leaf_dim)
            self.leaf_encoder = nn.Sequential(
                nn.Linear(n_trees * leaf_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        fusion_dim = hidden_dim
        if self.use_aux:
            fusion_dim += 32
        if self.use_leaf:
            fusion_dim += hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, dense_x, aux_x, leaf_ids):
        pieces = [self.dense_encoder(dense_x)]
        if self.use_aux:
            pieces.append(self.aux_encoder(aux_x))
        if self.use_leaf:
            leaf_vec = self.leaf_embedding(leaf_ids).reshape(leaf_ids.shape[0], -1)
            pieces.append(self.leaf_encoder(leaf_vec))
        return self.fusion(torch.cat(pieces, dim=1)).squeeze(1)


def scores_from_model(model, dense_x, aux_x, leaf_ids, device, batch_size):
    model.eval()
    loader = make_loader(dense_x, aux_x, leaf_ids, np.zeros(len(dense_x)), batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch_dense, batch_aux, batch_leaf, _ in loader:
            logits = model(batch_dense.to(device), batch_aux.to(device), batch_leaf.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def train_treecross(mode, data, args, device):
    model = TreeCrossMPD(
        dense_dim=data["x_train"].shape[1],
        aux_dim=data["aux_train"].shape[1],
        total_leaves=data["total_leaves"],
        n_trees=data["leaf_train"].shape[1],
        leaf_dim=args.leaf_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mode=mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_train = data["y_train_float"]
    pos_weight_value = max(1.0, float((len(y_train) - y_train.sum()) / max(1, y_train.sum())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    train_loader = make_loader(
        data["x_train"],
        data["aux_train"],
        data["leaf_train"],
        data["y_train_float"],
        args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for dense_x, aux_x, leaf_ids, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            dense_x = dense_x.to(device)
            aux_x = aux_x.to(device)
            leaf_ids = leaf_ids.to(device)
            y = y.to(device)
            loss = criterion(model(dense_x, aux_x, leaf_ids), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_score = scores_from_model(model, data["x_val"], data["aux_val"], data["leaf_val"], device, args.batch_size)
        threshold, val_f1 = best_threshold(data["y_val"], val_score)
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
    val_score = scores_from_model(model, data["x_val"], data["aux_val"], data["leaf_val"], device, args.batch_size)
    threshold, _ = best_threshold(data["y_val"], val_score)
    test_score = scores_from_model(model, data["x_test"], data["aux_test"], data["leaf_test"], device, args.batch_size)
    return (
        history,
        best_epoch,
        evaluate(data["y_val"], val_score, threshold),
        evaluate(data["y_test"], test_score, threshold),
        val_score,
        test_score,
    )


def add_predictions(split_df, setting_name, model_name, score, threshold):
    pred_df = split_df[["package_key", "package_name", "version", "label"]].copy()
    pred_df["setting"] = setting_name
    pred_df["model"] = model_name
    pred_df["score"] = score
    pred_df["prediction"] = (score >= threshold).astype(int)
    pred_df["error_type"] = np.where(
        (pred_df["label"].eq(0) & pred_df["prediction"].eq(1)),
        "false_positive",
        np.where((pred_df["label"].eq(1) & pred_df["prediction"].eq(0)), "false_negative", "correct"),
    )
    return pred_df


def run_setting(feature_path, args, device, out_dir):
    setting_name = Path(feature_path).stem
    setting_out = out_dir / setting_name
    setting_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_path)
    df = df[df["has_package_json"].eq(1)].copy()
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    columns = feature_columns(df)

    train, val, test = split_data(df, args.random_state, args.test_size, args.val_size)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_csv(setting_out / f"{split_name}.csv", index=False)

    x_train_raw = train[columns].fillna(0).astype(float)
    x_val_raw = val[columns].fillna(0).astype(float)
    x_test_raw = test[columns].fillna(0).astype(float)
    y_train = train["label"].to_numpy(dtype=np.int64)
    y_val = val["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)

    print(f"[{setting_name}] fitting RF/HGB teachers...")
    rf, hgb = fit_tree_teachers(x_train_raw, y_train, args.random_state, args.rf_trees)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)
    aux_train = teacher_aux_features(rf, hgb, x_train_raw)
    aux_val = teacher_aux_features(rf, hgb, x_val_raw)
    aux_test = teacher_aux_features(rf, hgb, x_test_raw)

    train_leaves = rf.apply(x_train_raw)
    leaf_maps, total_leaves = build_leaf_maps(train_leaves)
    leaf_train = map_leaves(train_leaves, leaf_maps)
    leaf_val = map_leaves(rf.apply(x_val_raw), leaf_maps)
    leaf_test = map_leaves(rf.apply(x_test_raw), leaf_maps)

    data = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "aux_train": aux_train,
        "aux_val": aux_val,
        "aux_test": aux_test,
        "leaf_train": leaf_train,
        "leaf_val": leaf_val,
        "leaf_test": leaf_test,
        "total_leaves": total_leaves,
        "y_train_float": y_train.astype(np.float32),
        "y_val": y_val,
        "y_test": y_test,
    }

    metrics = []
    predictions = []
    histories = {}
    val_scores = {}
    test_scores = {}
    for model_name, score_val, score_test in [
        ("rf_teacher", aux_val[:, 0], aux_test[:, 0]),
        ("hgb_teacher", aux_val[:, 1], aux_test[:, 1]),
        ("tree_teacher_avg", aux_val[:, 2], aux_test[:, 2]),
    ]:
        val_scores[model_name] = score_val
        test_scores[model_name] = score_test
        threshold, _ = best_threshold(y_val, score_val)
        metrics.append({"setting": setting_name, "model": model_name, "split": "val", "best_epoch": 0, **evaluate(y_val, score_val, threshold)})
        test_metrics = evaluate(y_test, score_test, threshold)
        metrics.append({"setting": setting_name, "model": model_name, "split": "test", "best_epoch": 0, **test_metrics})
        predictions.append(add_predictions(test, setting_name, model_name, score_test, test_metrics["threshold"]))
        fixed_val_metrics = evaluate(y_val, score_val, 0.5)
        fixed_test_metrics = evaluate(y_test, score_test, 0.5)
        fixed_name = f"{model_name}_fixed05"
        metrics.append({"setting": setting_name, "model": fixed_name, "split": "val", "best_epoch": 0, **fixed_val_metrics})
        metrics.append({"setting": setting_name, "model": fixed_name, "split": "test", "best_epoch": 0, **fixed_test_metrics})
        predictions.append(add_predictions(test, setting_name, fixed_name, score_test, 0.5))

    for mode in ["treecross_prior", "treecross_leaf", "treecross_full"]:
        print(f"[{setting_name}] training {mode} on {device}...")
        history, best_epoch, val_metrics, test_metrics, val_score, test_score = train_treecross(mode, data, args, device)
        histories[mode] = history
        val_scores[mode] = val_score
        test_scores[mode] = test_score
        metrics.append({"setting": setting_name, "model": mode, "split": "val", "best_epoch": best_epoch, **val_metrics})
        metrics.append({"setting": setting_name, "model": mode, "split": "test", "best_epoch": best_epoch, **test_metrics})
        predictions.append(add_predictions(test, setting_name, mode, test_score, test_metrics["threshold"]))

    blend_specs = [
        ("blend_hgb_treecross_full", "hgb_teacher", "treecross_full"),
        ("blend_hgb_treecross_leaf", "hgb_teacher", "treecross_leaf"),
        ("blend_avg_treecross_full", "tree_teacher_avg", "treecross_full"),
        ("blend_rf_hgb", "rf_teacher", "hgb_teacher"),
    ]
    for model_name, left, right in blend_specs:
        blend = choose_blend(y_val, val_scores[left], val_scores[right], args.blend_step)
        val_score = blend["alpha"] * val_scores[left] + (1.0 - blend["alpha"]) * val_scores[right]
        test_score = blend["alpha"] * test_scores[left] + (1.0 - blend["alpha"]) * test_scores[right]
        val_metrics = evaluate(y_val, val_score, blend["threshold"])
        test_metrics = evaluate(y_test, test_score, blend["threshold"])
        metrics.append(
            {
                "setting": setting_name,
                "model": model_name,
                "split": "val",
                "best_epoch": 0,
                "blend_alpha": blend["alpha"],
                **val_metrics,
            }
        )
        metrics.append(
            {
                "setting": setting_name,
                "model": model_name,
                "split": "test",
                "best_epoch": 0,
                "blend_alpha": blend["alpha"],
                **test_metrics,
            }
        )
        predictions.append(add_predictions(test, setting_name, model_name, test_score, test_metrics["threshold"]))

    gate_specs = [
        ("gate_avg_treecross_full", "tree_teacher_avg", "treecross_full"),
        ("gate_avg_treecross_prior", "tree_teacher_avg", "treecross_prior"),
        ("gate_hgb_treecross_full", "hgb_teacher", "treecross_full"),
        ("gate_hgb_treecross_prior", "hgb_teacher", "treecross_prior"),
        ("gate_avg_fixed_treecross_full", "tree_teacher_avg", "treecross_full"),
    ]
    for model_name, teacher_name, fallback_name in gate_specs:
        gate = choose_confidence_gate(
            y_val,
            val_scores[teacher_name],
            val_scores[fallback_name],
            args.blend_step,
        )
        val_score = apply_confidence_gate(
            val_scores[teacher_name],
            val_scores[fallback_name],
            gate["confidence_margin"],
        )
        test_score = apply_confidence_gate(
            test_scores[teacher_name],
            test_scores[fallback_name],
            gate["confidence_margin"],
        )
        val_metrics = evaluate(y_val, val_score, gate["threshold"])
        test_metrics = evaluate(y_test, test_score, gate["threshold"])
        metrics.append(
            {
                "setting": setting_name,
                "model": model_name,
                "split": "val",
                "best_epoch": 0,
                "confidence_margin": gate["confidence_margin"],
                **val_metrics,
            }
        )
        metrics.append(
            {
                "setting": setting_name,
                "model": model_name,
                "split": "test",
                "best_epoch": 0,
                "confidence_margin": gate["confidence_margin"],
                **test_metrics,
            }
        )
        predictions.append(add_predictions(test, setting_name, model_name, test_score, test_metrics["threshold"]))

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df.to_csv(setting_out / "metrics.csv", index=False)
    predictions_df.to_csv(setting_out / "predictions.csv", index=False)
    (setting_out / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")

    return {
        "setting": setting_name,
        "feature_source": feature_path,
        "rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "feature_count": int(len(columns)),
        "rf_trees": int(args.rf_trees),
        "total_rf_leaves": int(total_leaves),
        "features": columns,
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

    all_metrics = pd.concat([pd.read_csv(Path(s["outputs"]["metrics"])) for s in summaries], ignore_index=True)
    all_metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = {
        "model_family": args.model_family,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "args": vars(args),
        "settings": summaries,
        "metrics_csv": str(out_dir / "metrics.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(all_metrics.to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
