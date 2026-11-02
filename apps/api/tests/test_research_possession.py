"""Synthetic trajectories and reviewed-label contracts; no match captures."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from research_tracking import Capture, fill_short_gaps, load_capture, rts_smooth, time_segments
from research_events import ResearchConfig, infer, movement_events
from research_validation import match_records, validate_research
from research_control import FEATURES, probabilities, train_model


def capture(n=91):
    t = np.arange(n) / 30.
    xy = np.zeros((n, 3, 2), dtype=np.float32)
    xy[:, 1, 0], xy[:, 2, 0] = 10., 20.
    ball = np.zeros((n, 2), dtype=np.float32)
    return Capture(t, np.ones(n, dtype=np.int8), np.arange(n),
                   ['home:10', 'home:9', 'away:8'], xy, np.ones((n, 3)),
                   ball, np.zeros(n), np.ones(n), {})


class ResearchTrackingTests(unittest.TestCase):
    def test_short_gap_is_filled_but_long_gap_and_period_boundary_are_not(self):
        t = np.arange(9) / 10.
        x = np.array([0., np.nan, 2., np.nan, np.nan, np.nan, 6., np.nan, 8.])[:, None]
        p = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2])
        out, reconstructed = fill_short_gaps(t, p, x, .25)
        self.assertAlmostEqual(out[1, 0], 1.)
        self.assertTrue(reconstructed[1, 0])
        self.assertTrue(np.isnan(out[3:6]).all())
        self.assertTrue(np.isnan(out[7, 0]))

    def test_rts_preserves_linear_motion_and_does_not_bridge_missing_segments(self):
        t = np.arange(100) / 30.
        x = (2 * t)[:, None]
        x[40:60] = np.nan
        pos, vel = rts_smooth(t, np.ones(100), x, np.ones_like(x), process_noise=16.)
        self.assertTrue(np.isnan(pos[40:60]).all())
        self.assertLess(np.max(np.abs(pos[10:30, 0] - 2 * t[10:30])), .06)
        self.assertLess(abs(np.mean(vel[10:30, 0]) - 2.), .15)

    def test_loader_uses_estimated_coordinates_but_never_embedded_actors(self):
        base = dict(frameNum=1, period=1, videoTimeMs=0,
                    homePlayers=[dict(jerseyNum='10', x=0., y=0., confidence='LOW', visibility='ESTIMATED')],
                    awayPlayers=[], balls=[dict(x=0., y=0., z=0., visibility='ESTIMATED')])
        other = dict(base, frameNum=2, videoTimeMs=33.333)
        payload = '\n'.join(json.dumps(r) for r in [base, base, other]).encode('utf-8-sig')
        with patch.object(Path, 'open', return_value=io.BytesIO(payload)):
            a = load_capture(Path('fixture.jsonl'), 'raw')
        base['game_event'] = dict(home_ball=False, player_id='wrong')
        base['possession_event'] = dict(possession_event_type='SH')
        payload = '\n'.join(json.dumps(r) for r in [base, base, other]).encode('utf-8-sig')
        with patch.object(Path, 'open', return_value=io.BytesIO(payload)):
            b = load_capture(Path('fixture.jsonl'), 'raw')
        np.testing.assert_array_equal(a.xy, b.xy)
        self.assertEqual(a.keys, ['home:10'])
        self.assertGreater(a.player_quality[0, 0], 0.)
        self.assertLess(a.player_quality[0, 0], 1.)
        self.assertEqual(a.audit['duplicate_frames_skipped'], 1)

    def test_episode_segmentation_splits_periods_and_missing_time(self):
        t = np.array([0., .1, 1., 1.1, 1.2])
        p = np.array([1, 1, 1, 2, 2])
        self.assertEqual(time_segments(np.ones(5, dtype=bool), t, p), [(0, 2), (2, 3), (3, 5)])


def prepared(c):
    velocity = np.gradient(c.ball, c.time, axis=0).astype(np.float32)
    return dict(xy=c.xy.copy(), ball=c.ball.copy(), player_velocity=np.zeros_like(c.xy),
                ball_velocity=velocity, acceleration=np.gradient(velocity, c.time, axis=0),
                player_quality=c.player_quality, ball_quality=c.ball_quality,
                reconstructed_ball=np.zeros(len(c.time), dtype=bool))


class ResearchEventTests(unittest.TestCase):
    def test_estimated_positions_can_support_interaction_without_trained_control_claim(self):
        c = capture()
        c.player_quality[:] = .2
        c.ball_quality[:] = .2
        rows, touches, actions = infer(c, prepared(c))
        self.assertEqual(rows[20]['carrier'], 'home:10')
        self.assertEqual(rows[20]['team'], 'home')
        self.assertEqual(rows[20]['control_status'], 'likely_control')
        self.assertEqual(rows[20]['control_probability'], '')
        self.assertTrue(actions)

    def test_one_touch_does_not_require_duration_confirmation(self):
        c = capture()
        c.ball[:, 0] = 10 + 15 * (c.time - 1.)
        c.xy[:, 0, 0] = -30.
        p = prepared(c)
        p['acceleration'][:] = 0.
        p['acceleration'][30, 0] = 30.
        rows, touches, _ = infer(c, p)
        self.assertTrue(any(t['actor'] == 'home:9' and abs(t['time_s'] - 1.) < .1 for t in touches))
        self.assertEqual(rows[30]['carrier'], 'home:9')
        self.assertEqual(rows[30]['control_status'], 'interaction_only')

    def test_contest_keeps_previous_owner_as_history_but_does_not_assign_control(self):
        c = capture()
        c.xy[20:40, 2, 0] = .2
        c.ball[20:40, 0] = .1
        rows, _, _ = infer(c, prepared(c))
        self.assertEqual(rows[30]['status'], 'contested')
        self.assertEqual(rows[30]['carrier'], '')
        self.assertEqual(rows[30]['team'], '')
        self.assertEqual(rows[30]['previous_owner'], 'home:10')

    def test_missing_ball_and_period_reset_do_not_carry_possession_forward(self):
        c = capture()
        c.ball[40:60] = np.nan
        c.period[60:] = 2
        rows, _, _ = infer(c, prepared(c))
        self.assertEqual(rows[50]['team'], '')
        self.assertEqual(rows[50]['carrier'], '')
        self.assertEqual(rows[50]['status'], 'missing_ball')

    def test_touch_pass_reception_and_out_candidates_from_geometry(self):
        c = capture(121)
        c.ball[:, 0] = np.clip((c.time - 1.) * 10., 0., 10.)
        p = prepared(c)
        p['acceleration'][:] = 0.
        p['acceleration'][30, 0] = 20.
        p['acceleration'][60, 0] = -20.
        rows, touches, actions = infer(c, p)
        events = movement_events(c, p, rows, touches, actions, ResearchConfig())
        self.assertTrue(any(e['event_type'] == 'completed_pass_candidate' and
                            e['actor'] == 'home:10' and e['target'] == 'home:9' for e in events))

    def test_carry_and_pressure_candidates_have_duration_and_participants(self):
        c = capture()
        c.ball[:, 0] = c.time * 2
        c.xy[:, 0, 0] = c.ball[:, 0]
        c.xy[:, 2, 0] = c.ball[:, 0] + 2.
        p = prepared(c)
        p['player_velocity'][:, 0, 0] = 2.
        p['acceleration'][:] = 0.
        rows, touches, actions = infer(c, p)
        events = movement_events(c, p, rows, touches, actions, ResearchConfig())
        self.assertTrue(any(e['event_type'] == 'carry_candidate' for e in events))
        self.assertTrue(any(e['event_type'] == 'pressure_candidate' and e['target'] == 'away:8' for e in events))

    def test_matching_maximizes_matches_instead_of_greedily_losing_one(self):
        preds = [dict(time_s=t, period=1, identity='home:10') for t in [.4, .9]]
        refs = [dict(time_s=t, period=1, identity='home:10') for t in [0., .5]]
        report, matches = match_records(preds, refs, tolerance=.5)
        self.assertEqual(report['matched'], 2)
        self.assertEqual(len({m['reference_index'] for m in matches}), 2)
        self.assertEqual(match_records(preds, [dict(refs[0], period=2)], .5)[0]['matched'], 0)

    def test_sparse_touch_references_do_not_claim_precision(self):
        report, _ = match_records([dict(time_s=0., period=1)], [dict(time_s=0., period=1)], exhaustive=False)
        self.assertEqual(report['reference_recall'], 1.)
        self.assertIsNone(report['precision'])
        no_references, _ = match_records([dict(time_s=0., period=1)], [], exhaustive=True)
        self.assertIsNone(no_references['precision'])
        self.assertFalse(no_references['precision_supported'])

    def test_contest_validation_requires_both_participants_and_can_check_winner(self):
        c = capture()
        c.xy[20:40, 2, 0] = .2
        c.ball[20:40, 0] = .1
        p = prepared(c)
        rows, touches, actions = infer(c, p)
        predicted = movement_events(c, p, rows, touches, actions, ResearchConfig())
        reference = dict(startTime=.7, eventTime=1., gameEvents=dict(period=1),
                         possessionEvents=dict(possessionEventType='CH', carrierPlayerId=1,
                                               challengerPlayerId=3, challengeWinnerPlayerId=1),
                         homePlayers=[dict(playerId=1, jerseyNum=10)],
                         awayPlayers=[dict(playerId=3, jerseyNum=8)])
        report, _, matches = validate_research(rows, predicted, [reference])
        self.assertEqual(report['event_detection']['contest']['matched'], 1)
        self.assertTrue(next(m for m in matches if m['category'] == 'contest')['winner_matches'])
        self.assertEqual(report['event_detection']['contest']['winner_among_matched_contests']['correct'], 1)

    def test_corner_restart_is_not_misclassified_as_touchline_throw(self):
        c = capture()
        c.ball[:] = [52., 33.]
        c.xy[:, 0] = c.ball
        p = prepared(c)
        p['acceleration'][:] = 0.
        p['acceleration'][45, 0] = 10.
        rows, touches, actions = infer(c, p)
        events = movement_events(c, p, rows, touches, actions, ResearchConfig())
        self.assertTrue(any(e['event_type'] == 'corner_candidate' for e in events))
        self.assertFalse(any(e['event_type'] == 'throw_in_candidate' for e in events))

    def test_out_crossing_is_geometry_only_and_does_not_assign_possession(self):
        c = capture()
        c.ball[:, 0] = np.linspace(50., 55., len(c.time))
        p = prepared(c)
        rows, touches, actions = infer(c, p)
        events = movement_events(c, p, rows, touches, actions, ResearchConfig())
        self.assertTrue(any(e['event_type'] == 'out_candidate' for e in events))
        self.assertTrue(any(e['event_type'] == 'goal_line_crossing_candidate' for e in events))
        self.assertEqual(rows[-1]['team'], '')

    def test_pass_reference_cannot_match_anywhere_along_long_flight(self):
        candidate = dict(event_type='completed_pass_candidate', period=1, time_s=0., end_time_s=4.,
                         actor='home:10', target='home:9')
        reference = dict(eventTime=3., gameEvents=dict(period=1),
                         possessionEvents=dict(possessionEventType='PA', passerPlayerId=1,
                                               receiverPlayerId=2, passOutcomeType='C'),
                         homePlayers=[dict(playerId=1, jerseyNum=10), dict(playerId=2, jerseyNum=9)])
        report, _, _ = validate_research([], [candidate], [reference])
        self.assertEqual(report['event_detection']['completed_pass']['matched'], 0)


class ResearchControlTests(unittest.TestCase):
    def reviewed_records(self):
        records = []
        for split in ['train', 'test']:
            for label in [0, 1]:
                for i in range(3):
                    records.append(dict({f: float(label * 10 + i * .01) for f in FEATURES},
                                        reviewed='yes', label_control=str(label), split=split,
                                        capture_id=split, period=1, start_time_s=i * 10, end_time_s=i * 10 + 1))
        return records

    def test_classifier_requires_reviewed_labels_and_separate_test_data(self):
        with self.assertRaises(ValueError):
            train_model([], minimum_per_class=2)
        records = self.reviewed_records()
        model = train_model(records, minimum_per_class=2)
        scores = probabilities(model, [records[0], records[3]])
        self.assertLess(scores[0], .1)
        self.assertGreater(scores[1], .9)
        self.assertEqual(model['reviewed_test_result']['correct'], 6)
        records[-1]['capture_id'] = 'train'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            train_model(records, minimum_per_class=2)


if __name__ == '__main__':
    unittest.main()
