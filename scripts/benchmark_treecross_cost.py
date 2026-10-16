"""Measure one frozen split; retain paper accuracy runs unchanged."""
import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import train_treecross as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cscloud2026_latex/revision_20261012/performance'
SPLIT = ROOT / 'runs/treecross/random2000_v3_multiseed/seed_42/hard_case'


def timed(label, operation):
    start = time.perf_counter()
    value = operation()
    elapsed = time.perf_counter() - start
    print(f'{label}: {elapsed:.3f} s', flush=True)
    return value, elapsed


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((SPLIT.parent / 'summary.json').read_text(encoding='utf-8'))
    args = SimpleNamespace(**config['args'])
    original.set_seed(42)
    frames = {name: pd.read_csv(SPLIT / f'{name}.csv') for name in ('train', 'val', 'test')}
    columns = config['settings'][1]['features']
    x = {name: frame[columns].fillna(0).astype(float) for name, frame in frames.items()}
    y = {name: frame['label'].to_numpy(dtype=np.int64) for name, frame in frames.items()}
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, l2_regularization=.01, random_state=42)
    rf = RandomForestClassifier(n_estimators=args.rf_trees, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=1, random_state=42)
    _, hgb_seconds = timed('HGB fit', lambda: hgb.fit(x['train'], y['train']))
    _, rf_seconds = timed('RF fit', lambda: rf.fit(x['train'], y['train']))
    start = time.perf_counter()
    scaler = StandardScaler().fit(x['train'])
    maps, total_leaves = original.build_leaf_maps(rf.apply(x['train']))
    data = {'total_leaves': total_leaves, 'y_train_float': y['train'].astype(np.float32), 'y_val': y['val'], 'y_test': y['test']}
    for name in frames:
        data[f'x_{name}'] = scaler.transform(x[name]).astype(np.float32)
        data[f'aux_{name}'] = original.teacher_aux_features(rf, hgb, x[name])
        data[f'leaf_{name}'] = original.map_leaves(rf.apply(x[name]), maps)
    preprocessing_seconds = time.perf_counter() - start
    # Capture the trained module without changing the original training function.
    captured = []
    constructor = original.TreeCrossMPD
    def capture(*positional, **keywords):
        model = constructor(*positional, **keywords)
        captured.append(model)
        return model
    original.TreeCrossMPD = capture
    try:
        training, neural_seconds = timed('Full branch fit and validation', lambda: original.train_treecross('treecross_full', data, args, torch.device('cuda')))
    finally:
        original.TreeCrossMPD = constructor
    history, best_epoch, _, _, val_score, test_score = training
    gate, gate_seconds = timed('Gate/threshold validation search', lambda: original.choose_confidence_gate(y['val'], data['aux_val'][:, 1], val_score, args.blend_step))
    model = captured[0].cpu().eval()
    joblib.dump(hgb, OUT / 'hgb.joblib')
    joblib.dump({'hgb': hgb, 'rf': rf, 'scaler': scaler, 'maps': maps, 'total_leaves': total_leaves, 'columns': columns, 'args': vars(args), 'gate': gate}, OUT / 'full.joblib')
    torch.save(model.state_dict(), OUT / 'full_weights.pt')
    frozen = pd.read_csv(SPLIT / 'predictions.csv')
    old_teacher = frozen[frozen['model'].eq('hgb_teacher')].set_index('package_key').loc[frames['test']['package_key'], 'score'].to_numpy()
    rebuilt_teacher = hgb.predict_proba(x['test'])[:, 1]
    full_score = original.apply_confidence_gate(rebuilt_teacher, test_score, gate['confidence_margin'])
    result = {'seed':42, 'setting':'hard_case', 'train_rows':len(x['train']), 'val_rows':len(x['val']), 'test_rows':len(x['test']), 'cpu':platform.processor(), 'cpu_model':os.environ.get('BENCHMARK_CPU_MODEL', platform.processor()), 'gpu':torch.cuda.get_device_name(0), 'python':platform.python_version(), 'torch':torch.__version__, 'sklearn':sklearn.__version__, 'cpu_threads':1, 'training_neural_device':'cuda', 'hgb_fit_s':hgb_seconds, 'rf_fit_s':rf_seconds, 'representation_preparation_s':preprocessing_seconds, 'neural_fit_validation_s':neural_seconds, 'gate_search_s':gate_seconds, 'full_fit_validation_s':hgb_seconds+rf_seconds+preprocessing_seconds+neural_seconds+gate_seconds, 'epochs_run':len(history), 'best_epoch':best_epoch, 'gate':gate, 'teacher_score_max_difference_from_frozen':float(np.max(np.abs(old_teacher-rebuilt_teacher))), 'cost_run_f1':original.evaluate(y['test'], full_score, gate['threshold'])['f1'], 'accuracy_note':'Separate cost run: full branch is trained directly, not after the original prior/leaf ablation RNG sequence. It does not replace paper accuracy results.'}
    (OUT / 'training.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def measure(kind):
    device = torch.device('cuda' if kind == 'full_cuda' else 'cpu')
    frame = pd.read_csv(SPLIT / 'test.csv')
    config = json.loads((SPLIT.parent / 'summary.json').read_text(encoding='utf-8'))
    columns = config['settings'][1]['features']
    raw = frame[columns].fillna(0).astype(float)
    if kind == 'hgb_cpu':
        hgb = joblib.load(OUT / 'hgb.joblib')
        def predict(values):
            return hgb.predict_proba(values)[:, 1] >= .5
    else:
        bundle = joblib.load(OUT / 'full.joblib')
        args = bundle['args']
        model = original.TreeCrossMPD(len(columns), 4, bundle['total_leaves'], args['rf_trees'], args['leaf_dim'], args['hidden_dim'], args['dropout'], 'treecross_full')
        model.load_state_dict(torch.load(OUT / 'full_weights.pt', map_location='cpu', weights_only=True))
        model.to(device).eval()
        def predict(values):
            aux = original.teacher_aux_features(bundle['rf'], bundle['hgb'], values)
            leaves = original.map_leaves(bundle['rf'].apply(values), bundle['maps'])
            dense = bundle['scaler'].transform(values).astype(np.float32)
            score = original.scores_from_model(model, dense, aux, leaves, device, 128)
            return original.apply_confidence_gate(aux[:, 1], score, bundle['gate']['confidence_margin']) >= bundle['gate']['threshold']
    if device.type == 'cuda':
        torch.cuda.synchronize()
    for _ in range(5):
        predict(raw)
        predict(raw.iloc[:1])
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    peak = [initial_rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.005):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    timings = {}
    for name, values in [('batch338', raw), ('single', raw.iloc[:1])]:
        samples = []
        for _ in range(50):
            start = time.perf_counter()
            output = predict(values)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
            assert output.shape == (len(values),)
        timings[name] = {'median_ms':statistics.median(samples), 'p95_ms':float(np.quantile(samples,.95)), 'median_ms_per_package':statistics.median(samples)/len(values), 'repeats':50, 'samples_ms':samples}
    peak[0] = max(peak[0], process.memory_info().rss)
    stop.set()
    sampler.join()
    result = {'kind':kind, 'timings':timings, 'inference_process_peak_rss_mib':peak[0]/2**20, 'post_warmup_rss_mib':initial_rss/2**20, 'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0.0, 'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0.0, 'scope':'Feature-table input through classification, including teacher scoring, scaling, leaf mapping, neural data loading/transfers and gate. Excludes archive parsing, model loading and training. Absolute RSS includes Python/framework/data, measured in separate processes.'}
    (OUT / f'{kind}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='timings'}), flush=True)
    print({name:{k:v for k,v in t.items() if k!='samples_ms'} for name,t in timings.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','hgb_cpu','full_cpu','full_cuda'])
    selected = parser.parse_args().mode
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        prepare() if selected=='prepare' else measure(selected)

