"""Contract fixtures, not real match events: regressions for as-of-frame accounting."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import PitchGrid
from player_live import LivePlayers, PhysicalIndex


class LiveTimingTests(unittest.TestCase):
    def live(self):
        n = 240
        players = pd.DataFrame([
            dict(slot=0, player_id='1', jersey='10', name='Passer', side='home', team_name='Home', position=None),
            dict(slot=1, player_id='2', jersey='9', name='Receiver', side='home', team_name='Home', position=None),
        ])
        base = dict(ball_x=0., ball_y=0., end_ball_x=10., end_ball_y=0.,
                    setpiece_type='O', is_goal=False)
        events = pd.DataFrame([
            dict(base, frame_idx=30, frame_num=30, event_time_s=1., poss_type='PA',
                 passer_id='1', player_id='1', receiver_id='2', pass_outcome='C'),
            dict(base, frame_idx=180, frame_num=180, event_time_s=6., poss_type='SH',
                 shooter_id='2', player_id='2', shot_outcome='G', is_goal=True),
        ])
        match = SimpleNamespace(n_frames=n, grid=PitchGrid(), fps=30.,
                                players=players, events=events, X=np.zeros((n, 2)),
                                Y=np.zeros((n, 2)), SPEED=np.ones((n, 2)),
                                period=np.ones(n, dtype=int), match_minute=np.arange(n) / 1800,
                                attack_dir={1: {'home': 1, 'away': -1}})
        return LivePlayers(SimpleNamespace(m=match, action_values=lambda: pd.DataFrame()))

    def test_future_shot_does_not_credit_assist_or_key_pass_early(self):
        live = self.live()
        before = live.table(60)[0]
        self.assertEqual(before['assists'], 0)
        self.assertEqual(before['key_passes'], 0)
        self.assertEqual(before['xa'], 0.)
        after = live.table(180)[0]
        self.assertEqual(after['assists'], 1)
        self.assertEqual(after['key_passes'], 1)
        self.assertGreater(after['xa'], 0.)

    def test_pass_map_does_not_reveal_future_shot_labels(self):
        live = self.live()
        before = live._actions_at(60)
        maps = live._maps(before[before.slot == 0], 'home')
        self.assertFalse(maps['passes'][0]['key'])
        self.assertFalse(maps['passes'][0]['assist'])
        # Going back in time must not mutate the cached complete-match events.
        self.assertEqual(live.table(180)[0]['assists'], 1)
        self.assertEqual(live.table(60)[0]['assists'], 0)

    def test_rebound_shot_does_not_delay_or_overwrite_first_shot_credit(self):
        live = self.live()
        live.m.events.loc[1, ['shot_outcome', 'is_goal']] = ['S', False]
        later = live.m.events.iloc[1].copy()
        later['frame_idx'], later['frame_num'], later['event_time_s'] = 210, 210, 7.
        later['shot_outcome'], later['is_goal'] = 'G', True
        live.m.events = pd.concat([live.m.events, later.to_frame().T], ignore_index=True)
        self.assertEqual(live.table(180)[0]['key_passes'], 1)
        self.assertEqual(live.table(180)[0]['assists'], 0)
        self.assertEqual(live.table(210)[0]['assists'], 0)

    def physical(self, speed, period=None):
        s = np.asarray(speed, dtype=np.float32).reshape(-1, 1)
        return PhysicalIndex(SimpleNamespace(X=np.zeros_like(s), SPEED=s, fps=30.,
                                             period=np.ones(len(s)) if period is None else period))

    def test_sprint_counts_when_minimum_duration_is_met_even_if_run_continues(self):
        ph = self.physical([8.] * 120)
        self.assertEqual(int(ph.at(28)['sprints'][0]), 0)
        self.assertEqual(int(ph.at(29)['sprints'][0]), 1)
        self.assertEqual(int(ph.at(59)['sprints'][0]), 1)

    def test_sprint_count_is_independent_of_future_recording_length(self):
        self.assertEqual(int(self.physical([8.] * 60).at(59)['sprints'][0]),
                         int(self.physical([8.] * 120).at(59)['sprints'][0]))

    def test_fast_runs_do_not_join_across_halftime(self):
        ph = self.physical([8.] * 120, np.array([1] * 60 + [2] * 60))
        self.assertEqual(int(ph.at(119)['sprints'][0]), 2)


if __name__ == '__main__':
    unittest.main()
