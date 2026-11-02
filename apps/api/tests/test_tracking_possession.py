"""Synthetic movement fixtures; no private match data or event-fed predictions."""
import sys
import json
import io
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from tracking_possession import (PossessionTracker, detect_transfers, predict_file,
                                 summarize, validate_events, validate_transfers)


def frame(t=0., home=0., away=10., ball=0., z=0., period=1):
    return dict(frameNum=round(t * 30), videoTimeMs=t * 1000, period=period,
                homePlayers=[dict(jerseyNum='10', x=home, y=0., confidence='HIGH', visibility='VISIBLE')],
                awayPlayers=[dict(jerseyNum='10', x=away, y=0., confidence='HIGH', visibility='VISIBLE')],
                balls=[dict(x=ball, y=0., z=z, visibility='VISIBLE')])


class TrackingPossessionTests(unittest.TestCase):
    def tracker(self):
        return PossessionTracker(source='raw', confirm_s=0.)

    def test_identity_includes_team_and_ignores_embedded_labels(self):
        f = frame()
        f['game_event'] = dict(home_ball=False, player_id='wrong')
        f['possession_event'] = dict(possession_event_type='SH')
        self.assertEqual(self.tracker().update(f)['carrier'], 'home:10')
        self.assertEqual(self.tracker().update(frame(home=10., away=0.))['carrier'], 'away:10')

    def test_missing_airborne_and_contested_do_not_force_carrier(self):
        for f, status in [(frame(z=5.), 'airborne'), (frame(home=-.2, away=.2), 'contested')]:
            r = self.tracker().update(f)
            self.assertEqual(r['carrier'], '')
            self.assertEqual(r['status'], status)
        f = frame()
        f['balls'] = []
        self.assertEqual(self.tracker().update(f)['status'], 'missing_ball')

    def test_team_can_persist_in_flight_but_player_does_not(self):
        tr = self.tracker()
        tr.update(frame())
        r = tr.update(frame(t=.1, ball=3.))
        self.assertEqual(r['carrier'], '')
        self.assertEqual(r['team'], 'home')
        self.assertEqual(r['status'], 'in_flight_provisional')

    def test_period_and_gap_reset_previous_team(self):
        for f in [frame(t=10., ball=5.), frame(t=.1, ball=5., period=2)]:
            tr = self.tracker()
            tr.update(frame())
            self.assertEqual(tr.update(f)['team'], '')

    def test_contest_and_out_of_bounds_clear_team(self):
        for f in [frame(t=.1, home=2.8, away=3.2, ball=3.),
                  frame(t=.1, home=53., away=60., ball=53.)]:
            tr = self.tracker()
            tr.update(frame())
            self.assertEqual(tr.update(f)['team'], '')

    def test_confirmation_is_causal_and_resets_on_missing_ball(self):
        tr = PossessionTracker(source='raw', confirm_s=.1)
        self.assertEqual(tr.update(frame())['carrier'], '')
        self.assertEqual(tr.update(frame(t=.05))['carrier'], '')
        self.assertEqual(tr.update(frame(t=.1))['carrier'], 'home:10')
        f = frame(t=.15)
        f['balls'] = []
        tr.update(f)
        self.assertEqual(tr.update(frame(t=.2))['carrier'], '')

    def test_duration_is_timestamp_weighted_and_excludes_gaps(self):
        tr = self.tracker()
        rows = [tr.update(frame(t=t)) for t in [0., .05, .15, 10.]]
        s = summarize(rows)
        self.assertAlmostEqual(s['accounted_seconds'], .15)
        self.assertAlmostEqual(s['controlled_seconds']['home'], .15)

    def test_new_receiver_can_confirm_while_previous_team_is_provisional(self):
        tr = PossessionTracker(source='raw', confirm_s=.1)
        for t in [0., .05, .1]:
            tr.update(frame(t=t))
        for t in [.2, .25, .3]:
            f = frame(t=t, ball=3. + (t - .2) * 4.)
            f['homePlayers'].append(dict(jerseyNum='9', x=f['balls'][0]['x'], y=0.,
                                         confidence='HIGH', visibility='VISIBLE'))
            r = tr.update(f)
        self.assertEqual(r['carrier'], 'home:9')

    def test_validation_counts_abstentions_and_keeps_periods_separate(self):
        tr = self.tracker()
        rows = [tr.update(frame(t=0.)), tr.update(frame(t=.1, ball=3.))]
        events = [dict(eventTime=t, gameEvents=dict(period=p, playerId=7, homeTeam=True),
                       possessionEvents=dict(possessionEventType='PA', passerPlayerId=7),
                       homePlayers=[dict(playerId=7, jerseyNum=10)]) for t, p in [(0., 1), (.1, 1), (0., 2)]]
        report, details = validate_events(rows, events)
        self.assertEqual(report['all']['labels'], 3)
        self.assertEqual(report['all']['aligned'], 2)
        self.assertEqual(report['all']['covered'], 1)
        self.assertEqual(report['all']['correct_player'], 1)
        self.assertEqual(len(details), 3)

    def test_duplicate_frames_keep_first_and_report_conflicting_positions(self):
        a, b = frame(), frame()
        b['balls'][0]['x'] = 2.
        path = Path('synthetic-fixture.jsonl')
        payload = '\n'.join(json.dumps(f) for f in [a, b, frame(t=.1)]).encode('utf-8-sig')
        with patch.object(Path, 'open', return_value=io.BytesIO(payload)):
            rows, audit = predict_file(path, self.tracker())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['ball_x'], 0.)
        self.assertEqual(audit['duplicate_frames_skipped'], 1)
        self.assertEqual(audit['duplicate_frames_with_conflicting_tracking'], 1)

    def test_pass_and_recovery_candidates_require_observed_control(self):
        tr = self.tracker()
        rows = [tr.update(frame(away=5.))]
        f = frame(t=.1, home=4., away=5., ball=4.)
        f['homePlayers'][0]['jerseyNum'] = '9'
        rows.append(tr.update(f))
        rows.append(tr.update(frame(t=.2, home=0., away=5., ball=5.)))
        events = detect_transfers(rows)
        self.assertEqual([e['event_type'] for e in events], ['completed_pass_candidate', 'recovery_candidate'])
        rows[1]['status'], rows[1]['carrier'] = 'missing_ball', ''
        self.assertEqual(detect_transfers(rows), [])

    def test_pass_validation_is_one_to_one_and_requires_receiver(self):
        candidate = dict(event_type='completed_pass_candidate', period=1, release_time_s=0.,
                         from_player='home:10', to_player='home:9')
        reference = dict(eventTime=0., gameEvents=dict(period=1),
                         possessionEvents=dict(possessionEventType='PA', passOutcomeType='C',
                                               passerPlayerId=7, receiverPlayerId=8),
                         homePlayers=[dict(playerId=7, jerseyNum=10), dict(playerId=8, jerseyNum=9)])
        report, matches = validate_transfers([candidate, candidate], [reference])
        self.assertEqual(report['true_positive'], 1)
        self.assertEqual(report['false_positive'], 1)
        self.assertEqual(report['recall'], 1.)
        self.assertEqual(len(matches), 1)

    def test_transfer_does_not_bridge_unobserved_frame_gap(self):
        a = self.tracker().update(frame())
        b = dict(a, carrier='home:9', ball_x=5., video_time_s=1.)
        self.assertEqual(detect_transfers([a, b]), [])


if __name__ == '__main__':
    unittest.main()
