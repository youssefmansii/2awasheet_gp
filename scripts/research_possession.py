"""Run the research-informed, retrospective tracking-only possession/event model."""
import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research_tracking import load_capture, prepare
from research_events import ResearchConfig, infer, movement_events
from research_validation import validate_research
from tracking_possession import loads, summarize, player_labels, durations


def write_table(path, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def write_report(path, report):
    def percent(value):
        return f'{value * 100:.1f}%' if value is not None else 'not identifiable'
    summary, validation = report['summary'], report['validation']
    actor = validation['frame_actor_proxy']['all']
    lines = ['# Research-informed tracking-only analysis', '',
             f"Source: **{report['source']}**, {report['frames']:,} frames; retrospective RTS smoothing.", '',
             f"Control classifier: **{report['control_classifier']}**. No dense IBC accuracy is claimed.", '',
             '## Exact-time interaction actor proxy', '',
             f"Eligible event actors: {actor['labels']}; assigned: {actor['covered']}; correct: {actor['correct_player']}.",
             f"Coverage {percent(actor['carrier_coverage'])}; agreement when assigned {percent(actor['player_accuracy_when_covered'])}.",
             'A touch at release and sustained control are different targets. Event actors are sparse proxy labels.', '',
             '## Event validation', '',
             '| Category | Candidates | References | Matched | Precision | Reference recall | F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, values in validation['event_detection'].items():
        lines.append(f"| {name} | {values['predictions']} | {values['references']} | {values['matched']} | "
                     f"{percent(values['precision'])} | {percent(values['reference_recall'])} | {percent(values['f1'])} |")
    lines += ['', 'Touch/kick actor and carry reference lists are sparse. Unmatched candidates cannot all be treated as false positives.',
              'Passes require the same sender, receiver and period, with release time within 1 s. Other matching uses documented category tolerances.',
              'Contest matching requires both participants. Subtype names such as tackle/interception are not independently verified.', '',
              '## Inferred team possession', '',
              'Assigned time covers ' + percent(summary['team_assignment_coverage_percent'] / 100
                  if summary['team_assignment_coverage_percent'] is not None else None) + ' of accounted tracking time.',
              'Home/away share of assigned time only: ' + ' / '.join(
                  percent(summary['assigned_share_percent'][side] / 100 if summary['assigned_share_percent'][side] is not None else None)
                  for side in ['home', 'away']) + '.',
              '**This is not official whole-match possession.** There is no validated continuous ball-in-play flag; in-bounds stoppages may be included.', '',
              '## Candidate counts', '']
    lines += [f'- {key}: {value}' for key, value in report['candidate_counts'].items()]
    lines += ['', '## Limits', ''] + [f'- {line}' for line in report['limitations']]
    lines += ['- Shot/cross/clearance and contest precision may be poor; inspect each category above before using it.',
              '- No official goal, foul, card, offside or VAR decision is inferred.',
              '- These are one-match results, not held-out cross-match accuracy.', '',
              '## Files', '',
              '- `frames.csv`: physical interaction, heuristic/learned control state, provisional team possession and quality.',
              '- `predicted_events.csv`: candidate events, times, participants and evidence.',
              '- `event_matches.csv`, `actor_validation.csv`: auditable validation joins.',
              '- `control_annotations.csv`: blank reviewed control labels and train/test split for manual annotation.',
              '- `players.csv`: names and partial interaction totals.',
              '- `report.json`: parameters, quality audit and per-period validation.',
              '- `trajectories.npz`: processed positions and velocities; no event labels.', '',
              'Method source: [Link & Hoernig (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/).',
              'This is an adaptation with explicit extensions, not a reproduction of their trained K2 network.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tracking', type=Path, required=True)
    ap.add_argument('--events', type=Path, help='Optional provider events, used only for validation')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source', choices=['raw', 'smoothed'], default='raw')
    ap.add_argument('--control-model', type=Path)
    args = ap.parse_args()
    cfg = ResearchConfig()
    print('Reading tracking-only inputs', flush=True)
    c = load_capture(args.tracking, args.source)
    print(f'{len(c.time):,} distinct frames, {len(c.keys)} players', flush=True)
    p = prepare(c)
    print('Inferring touches, action intervals and provisional possession', flush=True)
    rows, touches, actions = infer(c, p, cfg)
    control_status = 'untrained; heuristic likely_control only'
    if args.control_model:
        from research_control import apply_model
        apply_model(rows, actions, loads(args.control_model.read_bytes()))
        control_status = 'Gaussian naive Bayes adaptation loaded from explicit reviewed labels'
    predicted = movement_events(c, p, rows, touches, actions, cfg)
    print('Predictions frozen. Validating supplied events.' if args.events else 'Predictions frozen. No validation events supplied.', flush=True)
    events = loads(args.events.read_bytes().removeprefix(b'\xef\xbb\xbf')) if args.events else []
    if not isinstance(events, list):
        raise ValueError('Validation events must be a JSON array')
    validation, actor_details, matches = validate_research(rows, predicted, events)
    validation['available'] = args.events is not None and any(v['references'] for v in validation['event_detection'].values())
    if not validation['available']:
        for values in validation['event_detection'].values():
            for result in [values, *values['by_period'].values()]:
                result.update(precision=None, reference_recall=None, f1=None, precision_supported=False,
                              warning='No events file supplied; accuracy has not been measured.')
    summary = summarize(rows)
    # The baseline summary's carrier time is interaction time in this model.
    summary['player_interaction_seconds'] = summary.pop('player_controlled_seconds')
    summary['interaction_seconds'] = summary.pop('controlled_seconds')
    control_time = Counter()
    for row, dt in zip(rows, durations(rows)):
        control_time[row['control_status']] += dt
    summary['control_estimate_seconds'] = dict(control_time)
    summary['warning'] = 'Inferred tracked-time share includes possible in-bounds stoppages; not official possession. Carrier means physical interaction, not validated IBC.'
    report = dict(method='Link & Hoernig 2017 inspired adaptation', source=args.source,
                  tracking_file=args.tracking.name, events_file=args.events.name if args.events else None,
                  retrospective=True, parameters=asdict(cfg), audit=c.audit,
                  trajectory_parameters=dict(short_gap_max_s=.3, player_process_noise=16.,
                                             ball_process_noise=400., measurement_sd_m=.3),
                  control_classifier=control_status, frames=len(rows), actions=len(actions),
                  candidate_counts=dict(Counter(e['event_type'] for e in predicted)),
                  summary=summary, validation=validation,
                  limitations=['No verified continuous ball-in-play flag.',
                               'Paper distance thresholds/K2 network unavailable; defaults are uncalibrated adaptations.',
                               'RTS uses future frames; provider smoothing provenance unknown.',
                               'Missing long trajectories stay unknown; candidates do not establish referee decisions.'])
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output / 'frames.csv', rows)
    write_table(args.output / 'predicted_events.csv', predicted)
    write_table(args.output / 'event_matches.csv', matches)
    write_table(args.output / 'actor_validation.csv', actor_details)
    features = [dict(a['features'], capture_id=args.tracking.name, reviewed='', label_control='', split='') for a in actions]
    write_table(args.output / 'control_annotations.csv', features)
    identities = player_labels(events)
    write_table(args.output / 'players.csv', [dict(player=k, name=identities.get(k, {}).get('name', ''),
                interaction_seconds=v) for k, v in sorted(summary['player_interaction_seconds'].items(), key=lambda kv: -kv[1])])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    write_report(args.output / 'REPORT.md', report)
    # Compact reusable trajectory cache contains predictions only, never event labels.
    np.savez_compressed(args.output / 'trajectories.npz', time=c.time, period=c.period, frame=c.frame,
                        keys=np.array(c.keys), ball=p['ball'], xy=p['xy'],
                        ball_velocity=p['ball_velocity'], player_velocity=p['player_velocity'],
                        acceleration=p['acceleration'], player_quality=p['player_quality'],
                        ball_quality=p['ball_quality'], height=c.height,
                        reconstructed_ball=p['reconstructed_ball'])
    print(json.dumps(dict(frame_actor_proxy=validation['frame_actor_proxy']['all'],
                         completed_pass=validation['event_detection']['completed_pass'],
                         candidates=report['candidate_counts']), indent=2))
    print(f'Wrote {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
