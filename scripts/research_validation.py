"""Post-prediction, period/participant-aware event validation."""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking_possession import ACTOR_FIELDS, event_roster, number, validate_events


def match_records(predictions, references, tolerance=.6, exhaustive=True):
    """Maximum-cardinality one-to-one match, then minimize absolute timing error.

    Candidate/reference intervals may overlap; participant 'identity' and period
    must agree. Dummy-free rectangular assignment is safe because invalid edges
    cost more than every possible valid edge combined.
    """
    groups_p, groups_r = defaultdict(list), defaultdict(list)
    for i, row in enumerate(predictions):
        groups_p[(row['period'], row.get('identity', ''))].append(i)
    for i, row in enumerate(references):
        groups_r[(row['period'], row.get('identity', ''))].append(i)
    matches = []
    for group, pi in groups_p.items():
        ri = groups_r.get(group, [])
        if not ri:
            continue
        pstart = np.array([predictions[i]['time_s'] for i in pi])[:, None]
        pend = np.array([predictions[i].get('end_time_s', predictions[i]['time_s']) for i in pi])[:, None]
        rstart = np.array([references[i]['time_s'] for i in ri])[None, :]
        rend = np.array([references[i].get('end_time_s', references[i]['time_s']) for i in ri])[None, :]
        error = np.maximum(np.maximum(pstart - rend, rstart - pend), 0.)
        allowed = error <= tolerance
        cost = np.where(allowed, error / max(tolerance, 1e-9), len(pi) + len(ri) + 1.)
        aa, bb = linear_sum_assignment(cost)
        for a, b in zip(aa, bb):
            if allowed[a, b]:
                matches.append(dict(prediction_index=pi[a], reference_index=ri[b],
                                    timing_error_s=float(error[a, b])))
    npred, nref, tp = len(predictions), len(references), len(matches)
    report = dict(predictions=npred, references=nref, matched=tp,
                  unmatched_predictions=npred - tp, missed_references=nref - tp,
                  reference_recall=tp / nref if nref else None,
                  precision=tp / npred if npred and nref and exhaustive else None,
                  f1=2 * tp / (npred + nref) if nref and exhaustive else None,
                  tolerance_s=tolerance,
                  precision_supported=bool(exhaustive and nref),
                  mean_timing_error_s=float(np.mean([m['timing_error_s'] for m in matches])) if matches else None)
    return report, matches


def validate_research(rows, predictions, events):
    roster, conflicts = event_roster(events)
    frames, details = validate_events(rows, events)
    references = defaultdict(list)
    for i, e in enumerate(events):
        ge, pe = e.get('gameEvents') or {}, e.get('possessionEvents') or {}
        if pe.get('nonEvent') is True or ge.get('videoMissing') is True:
            continue
        typ, period, t = pe.get('possessionEventType'), ge.get('period'), number(e.get('eventTime'))
        if not np.isfinite(t):
            continue
        actor_id = ge.get('playerId') if typ == 'IT' else pe.get(ACTOR_FIELDS.get(typ, ''))
        actor = roster.get(str(actor_id), '')
        base = dict(period=period, time_s=t, event_index=i, identity=actor)
        if actor and typ in ACTOR_FIELDS:
            references['touch_actor_proxy'].append(base)
            if typ in ('PA', 'CR', 'SH', 'CL'):
                references['kick_actor_proxy'].append(base)
            if typ in ('SH', 'CR', 'CL', 'BC'):
                group = {'SH': 'shot', 'CR': 'cross', 'CL': 'clearance', 'BC': 'carry'}[typ]
                references[group].append(base)
            if typ in ('PA', 'CR') and pe.get('passOutcomeType' if typ == 'PA' else 'crossOutcomeType') == 'C':
                receiver = roster.get(str(pe.get('receiverPlayerId')))
                if receiver:
                    references['completed_pass'].append(dict(base, identity=actor + '>' + receiver))
            elif typ in ('PA', 'CR') and pe.get('passOutcomeType' if typ == 'PA' else 'crossOutcomeType'):
                references['unsuccessful_pass'].append(base)
        if ge.get('gameEventType') == 'OUT':
            references['out'].append(dict(base, identity=''))
        if typ == 'CH':
            carrier = roster.get(str(pe.get('carrierPlayerId') or pe.get('ballCarrierPlayerId')))
            challenger = roster.get(str(pe.get('challengerPlayerId')))
            if not carrier or not challenger:
                carrier = roster.get(str(pe.get('homeDuelPlayerId')))
                challenger = roster.get(str(pe.get('awayDuelPlayerId')))
            if carrier and challenger:
                start = number(e.get('startTime'), t)
                references['contest'].append(dict(base, time_s=min(start, t), end_time_s=t,
                                                   identity='|'.join(sorted([carrier, challenger])),
                                                   winner=roster.get(str(pe.get('challengeWinnerPlayerId')), '')))
    specs = dict(touch_actor_proxy=('touch_candidate', .6, False),
                 kick_actor_proxy=('kick_candidate', .6, False),
                 completed_pass=('completed_pass_candidate', 1., True),
                 unsuccessful_pass=('unsuccessful_pass_candidate', 1., True),
                 shot=('shot_candidate', .6, True), cross=('cross_candidate', .6, True),
                 clearance=('clearance_candidate', .6, True),
                 carry=('carry_candidate', .6, False), contest=('contest_candidate', .6, True),
                 out=('out_candidate', 1., True))
    reports, all_matches = {}, []
    for name, (kind, tolerance, exhaustive) in specs.items():
        selected = []
        for i, e in enumerate(predictions):
            if e['event_type'] != kind:
                continue
            identity = e['actor']
            if name == 'completed_pass':
                identity += '>' + e['target']
            elif name == 'contest':
                identity = '|'.join(sorted([e['actor'], e.get('challenger', '')]))
            elif name == 'out':
                identity = ''
            # EventTime marks the release for passes, rather than the whole flight.
            candidate = dict(e, identity=identity, global_index=i)
            if name in ('completed_pass', 'unsuccessful_pass', 'cross'):
                candidate['end_time_s'] = candidate['time_s']
            selected.append(candidate)
        refs = references[name]
        report, matches = match_records(selected, refs, tolerance, exhaustive)
        report['by_period'] = {}
        for period in sorted({r['period'] for r in refs} | {r['period'] for r in selected}):
            report['by_period'][str(period)] = match_records(
                [r for r in selected if r['period'] == period],
                [r for r in refs if r['period'] == period], tolerance, exhaustive)[0]
        for m in matches:
            pred, ref = selected[m['prediction_index']], refs[m['reference_index']]
            all_matches.append(dict(category=name, prediction_index=pred['global_index'],
                                    event_index=ref['event_index'], timing_error_s=m['timing_error_s'],
                                    winner_matches=(pred.get('target') == ref.get('winner')) if ref.get('winner') else ''))
        if name == 'contest':
            known = [(selected[m['prediction_index']], refs[m['reference_index']]) for m in matches
                     if refs[m['reference_index']].get('winner')]
            assigned = [(pred, ref) for pred, ref in known if pred.get('target')]
            correct = sum(pred['target'] == ref['winner'] for pred, ref in assigned)
            report['winner_among_matched_contests'] = dict(labels=len(known), assigned=len(assigned), correct=correct,
                accuracy_when_assigned=correct / len(assigned) if assigned else None)
        if not exhaustive:
            report['warning'] = 'Sparse actor/carry labels are not an exhaustive touch/control inventory; precision is not identifiable.'
        reports[name] = report
    return dict(frame_actor_proxy=frames, event_detection=reports, identity_conflicts=conflicts,
                unvalidated=['dense_control', 'whole_match_possession', 'reception_timing',
                             'tackle_vs_interception_subtypes', 'pressure', 'restart_subtypes', 'official_goals'],
                warning='One-match retrospective evaluation; sparse actors are not dense control truth. No threshold tuning or held-out accuracy claim.'), details, all_matches
