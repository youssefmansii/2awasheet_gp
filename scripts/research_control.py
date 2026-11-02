"""Optional reviewed-label Gaussian Bayesian control classifier.

This is an explicit adaptation, not the paper's trained K2 Bayesian network.
No event-derived pseudo-labels or default trained probabilities are supplied.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


FEATURES = ['duration_s', 'mean_ball_speed', 'var_ball_speed', 'mean_ball_acceleration',
            'var_ball_acceleration', 'mean_ball_distance'] + [f'opponents_within_{r:.1f}m' for r in np.arange(.5, 5.1, .5)]


def probabilities(model, records):
    if model.get('version') != 1 or model.get('features') != FEATURES:
        raise ValueError('Unsupported control model or feature contract')
    if not records:
        return np.array([])
    x = np.array([[float(r[f]) for f in FEATURES] for r in records])
    if not np.isfinite(x).all():
        raise ValueError('Control features must be finite')
    means, variances, priors = [np.asarray(model[k], dtype=float) for k in ['means', 'variances', 'priors']]
    if (means.shape != (2, len(FEATURES)) or variances.shape != means.shape or priors.shape != (2,)
            or not np.isfinite(means).all() or not np.isfinite(variances).all()
            or not np.isfinite(priors).all() or (variances <= 0).any() or (priors <= 0).any()):
        raise ValueError('Invalid Gaussian control model parameters')
    logp = -.5 * np.sum(np.log(2 * np.pi * variances)[None] +
                        (x[:, None, :] - means[None]) ** 2 / variances[None], axis=-1) + np.log(priors)
    logp -= logp.max(axis=1, keepdims=True)
    weights = np.exp(logp)
    return weights[:, 1] / weights.sum(axis=1)


def train_model(records, minimum_per_class=20):
    selected = [r for r in records if str(r.get('reviewed', '')).lower() == 'yes']
    if any(str(r.get('label_control')) not in ('0', '1') or r.get('split') not in ('train', 'test') for r in selected):
        raise ValueError('Reviewed labels require label_control 0/1 and split train/test')
    train = [r for r in selected if r['split'] == 'train']
    test = [r for r in selected if r['split'] == 'test']
    for label in [0, 1]:
        if sum(int(r['label_control']) == label for r in train) < minimum_per_class:
            raise ValueError(f'Need at least {minimum_per_class} reviewed TRAIN examples of each class')
        if sum(int(r['label_control']) == label for r in test) < max(1, minimum_per_class // 4):
            raise ValueError('Need separately reviewed TEST examples of each class')
    if any(not r.get('capture_id') for r in selected):
        raise ValueError('Each reviewed row requires capture_id for overlap checking')
    for a in train:
        for b in test:
            if a['capture_id'] == b['capture_id'] and str(a['period']) == str(b['period']):
                gap = max(float(a['start_time_s']) - float(b['end_time_s']),
                          float(b['start_time_s']) - float(a['end_time_s']))
                if gap < 1.:
                    raise ValueError('TRAIN and TEST intervals overlap or are less than one second apart')
    x = np.array([[float(r[f]) for f in FEATURES] for r in train])
    if not np.isfinite(x).all():
        raise ValueError('Training features must be finite')
    labels = np.array([int(r['label_control']) for r in train])
    floor = np.maximum(np.var(x, axis=0) * .01, 1e-4)
    model = dict(version=1, method='reviewed-label Gaussian naive Bayes adaptation; not paper K2',
                 features=FEATURES, means=[x[labels == j].mean(axis=0).tolist() for j in [0, 1]],
                 variances=[np.maximum(x[labels == j].var(axis=0), floor).tolist() for j in [0, 1]],
                 priors=[float(np.mean(labels == j)) for j in [0, 1]],
                 training_examples=len(train), test_examples=len(test))
    pred = probabilities(model, test) >= .5
    actual = np.array([int(r['label_control']) for r in test])
    model['reviewed_test_result'] = dict(correct=int((pred == actual).sum()), total=len(test),
                                        confusion=[[int(((actual == a) & (pred == b)).sum()) for b in [0, 1]] for a in [0, 1]])
    return model


def apply_model(rows, actions, model):
    scores = probabilities(model, [a['features'] for a in actions])
    for action, score in zip(actions, scores):
        for r in rows[action['start_index']:action['end_index']]:
            if r['carrier'] == action['features']['player']:
                r['control_probability'] = float(score)
                r['control_status'] = 'model_control' if score >= .5 else 'model_interaction_without_control'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with args.annotations.open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    model = train_model(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(model['reviewed_test_result'], indent=2))


if __name__ == '__main__':
    main()
