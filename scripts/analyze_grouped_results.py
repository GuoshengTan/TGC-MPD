"""Audit grouped results against saved partitions and summarize fixed comparisons."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/treecross/package_name_grouped_20261012'
ORIGINAL = ROOT / 'runs/treecross/random2000_v3_multiseed'
MODELS = ['hgb_teacher_fixed05', 'hgb_teacher', 'gate_hgb_treecross_full']


def main():
    audit_rows = []
    for seed in (40, 41, 42):
        for setting in ('scoped_balanced', 'hard_case'):
            folder = RUN / f'seed_{seed}' / setting
            splits = {name: pd.read_csv(folder / f'{name}.csv') for name in ('train', 'val', 'test')}
            names = {name: set(frame.package_name) for name, frame in splits.items()}
            assert all(not names[a] & names[b] for a, b in [('train','val'),('train','test'),('val','test')])
            all_rows = pd.concat(splits.values())
            source = pd.read_csv(ROOT / f'data/processed/eval_settings/random2000/{setting}.csv')
            assert all_rows.package_key.is_unique
            assert set(all_rows.package_key) == set(source.package_key)
            expected = source.set_index('package_key').label
            observed = all_rows.set_index('package_key').label
            assert expected.sort_index().equals(observed.sort_index())
            metrics = pd.read_csv(folder / 'metrics.csv')
            metrics = metrics[metrics.split.eq('test')].set_index('model')
            predictions = pd.read_csv(folder / 'predictions.csv')
            for model in MODELS:
                rows = predictions[predictions.model.eq(model)].set_index('package_key')
                truth = splits['test'].set_index('package_key').label
                assert set(rows.index) == set(truth.index)
                assert rows.label.sort_index().equals(truth.sort_index())
                assert abs(f1_score(rows.label, rows.prediction) - metrics.loc[model, 'f1']) < 1e-12
            old_names = {name:set(pd.read_csv(ORIGINAL / f'seed_{seed}' / setting / f'{name}.csv').package_name) for name in ('train','val','test')}
            audit_rows.append({'seed':seed, 'setting':setting, 'train_rows':len(splits['train']), 'val_rows':len(splits['val']), 'test_rows':len(splits['test']), 'train_test_overlap_names':0, 'original_train_test_overlap_names':len(old_names['train'] & old_names['test']), 'teacher_threshold':float(metrics.loc['hgb_teacher','threshold']), 'gate_threshold':float(metrics.loc['gate_hgb_treecross_full','threshold']), 'gate_margin':float(metrics.loc['gate_hgb_treecross_full','confidence_margin'])})
    pd.DataFrame(audit_rows).to_csv(RUN / 'verified_split_audit.csv', index=False)
    summary = pd.read_csv(RUN / 'summary.csv')
    selected = summary[summary.model.isin(MODELS)].copy()
    selected.to_csv(RUN / 'paper_comparison.csv', index=False)
    original = pd.read_csv(ORIGINAL / 'summary.csv')
    lines = ['# 包名分组实验结果', '', '同一个包名的所有版本被限制在一个集合中。所有六次划分已检查：包名无交集，样本覆盖完整，标签一致，预测F1与指标文件一致。原来的包版本划分结果没有被覆盖。', '', '模型配置、三个种子和验证集选择规则保持不变。主要比较HGB teacher与预先指定的完整gate_hgb_treecross_full，两者均选择验证集阈值。']
    for setting in ('hard_case','scoped_balanced'):
        lines.extend(['', f'## {setting}', '', '| 场景 | 方法 | F1均值 ± 标准差 | Precision | Recall | PR-AUC |', '|---|---|---:|---:|---:|---:|'])
        for model in MODELS:
            r = selected[(selected.setting.eq(setting)) & (selected.model.eq(model))].iloc[0]
            lines.append(f'| {setting} | {model} | {r.f1_mean:.4f} ± {r.f1_std:.4f} | {r.precision_mean:.4f} | {r.recall_mean:.4f} | {r.pr_auc_mean:.4f} |')
        hgb = selected[(selected.setting.eq(setting)) & selected.model.eq('hgb_teacher')].iloc[0]
        gate = selected[(selected.setting.eq(setting)) & selected.model.eq('gate_hgb_treecross_full')].iloc[0]
        lines.extend(['', f'{setting}：完整模型相对验证集调阈值HGB的平均F1差值为{gate.f1_mean-hgb.f1_mean:+.6f}。这是本次分组划分上的比较，不是对原划分的配对差值。', ''])
    lines.extend(['## 原划分与新划分', '', '| 场景 | 原调阈值HGB F1 | 原完整模型F1 | 分组HGB F1 | 分组完整模型F1 |', '|---|---:|---:|---:|---:|'])
    for setting in ('hard_case','scoped_balanced'):
        def value(frame, model):
            return frame[frame.setting.eq(setting) & frame.model.eq(model)].iloc[0].f1_mean
        lines.append(f'| {setting} | {value(original,"hgb_teacher"):.4f} | {value(original,"gate_hgb_treecross_full"):.4f} | {value(selected,"hgb_teacher"):.4f} | {value(selected,"gate_hgb_treecross_full"):.4f} |')
    lines.extend(['', '本实验仅检验现有构造数据内的包名泛化，不等同于时间泛化、作者/发布者家族泛化或真实部署比例评价。三种子均值和标准差不自动证明统计显著性。', '', '协议见docs/package_name_grouped_protocol.md。所有划分CSV、预测和指标保留在本目录。'])
    (RUN / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(selected[['setting','model','f1_mean','f1_std','precision_mean','recall_mean','pr_auc_mean']].to_string(index=False))
    print(json.dumps(audit_rows, indent=2))


if __name__ == '__main__':
    main()

