"""Demo transport and artifact contracts, using synthetic local outputs."""
import csv
import json
import sys
import tempfile
import unittest
import asyncio
from pathlib import Path

import httpx
import numpy as np
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_demo import DemoResult, JobStore, create_app


def fixture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(source='raw', frames=2, control_classifier='untrained', candidate_counts={'touch_candidate': 1},
                  summary=dict(accounted_seconds=.1, team_assignment_coverage_percent=50.,
                               assigned_share_percent={'home': 100., 'away': 0.}, player_interaction_seconds={'home:10': .1}),
                  validation={'available': False})
    (folder / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    tables = {
        'frames.csv': [dict(video_time_s=1., frame_num=30, period=1, carrier='home:10', team='home',
                            status='likely_control', control_status='likely_control', previous_owner='',
                            tracking_quality=.6, reconstructed_ball=False, control_probability=''),
                       dict(video_time_s=1.1, frame_num=33, period=1, carrier='', team='', status='missing_ball',
                            control_status='unknown', previous_owner='', tracking_quality=0., reconstructed_ball=False, control_probability='')],
        'predicted_events.csv': [dict(event_type='touch_candidate', period=1, time_s=1., end_time_s=1.,
                                      actor='home:10', target='', evidence='stable_proximity')],
        'players.csv': [dict(player='home:10', name='Fixture Player', interaction_seconds=.1)]}
    for name, rows in tables.items():
        with (folder / name).open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(folder / 'trajectories.npz', time=[1., 1.1], period=[1, 1], frame=[30, 33],
                        keys=['home:10'], xy=np.array([[[0., 0.]], [[1., 0.]]]),
                        ball=[[0., 0.], [np.nan, np.nan]], height=[0., np.nan],
                        player_velocity=np.zeros((2, 1, 2)), ball_velocity=np.zeros((2, 2)))


class DemoResultTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[3] / 'tmp' / 'possession-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.path = Path(self.tmp.name)
        fixture(self.path / 'sample')

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_ball_is_null_and_tracking_only_has_no_accuracy(self):
        result = DemoResult(self.path / 'sample')
        self.assertIsNone(result.frame_at(1.1)['ball'])
        self.assertIsNone(result.manifest()['validation'])
        self.assertEqual(result.frame_at(1.)['players'][0]['name'], 'Fixture Player')
        self.assertEqual(result.events_page('touch_candidate', 'fixture', 0, 50)['total'], 1)
        self.assertEqual(result.events_page('shot_candidate', '', 0, 50)['total'], 0)

    def test_busy_jobs_reject_replacement_and_cancel_preserves_sample(self):
        store = JobStore(self.path / 'jobs', self.path / 'sample')
        job = store.reserve('first.jsonl', 'raw')
        with self.assertRaises(HTTPException) as rejected:
            store.reserve('second.jsonl', 'raw')
        self.assertEqual(rejected.exception.status_code, 409)
        store.cancel(job.id)
        store.reserve('next.jsonl', 'raw')
        self.assertTrue((self.path / 'sample' / 'report.json').exists())


class DemoApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_tracking_only_upload_runs_model_and_serves_replay(self):
        parent = Path(__file__).resolve().parents[3] / 'tmp' / 'possession-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            path = Path(tmp)
            store = JobStore(path / 'jobs', path / 'absent-sample')
            transport = httpx.ASGITransport(app=create_app(store), client=('127.0.0.1', 1234))
            lines = []
            for i in range(91):
                lines.append(json.dumps(dict(frameNum=i, period=1, videoTimeMs=i / 30 * 1000,
                    homePlayers=[dict(jerseyNum='10', x=0., y=0., confidence='HIGH', visibility='VISIBLE')],
                    awayPlayers=[dict(jerseyNum='8', x=15., y=0., confidence='HIGH', visibility='VISIBLE')],
                    balls=[dict(x=0., y=0., z=0., visibility='VISIBLE')])))
            async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1:8010') as api:
                r = await api.post('/api/possession/jobs', json={'name': 'synthetic.jsonl', 'source': 'raw'})
                job_id = r.json()['id']
                r = await api.post(f'/api/possession/jobs/{job_id}/files', files={'tracking': ('synthetic.jsonl', '\n'.join(lines).encode(), 'application/x-ndjson')})
                self.assertEqual(r.status_code, 200)
                for _ in range(150):
                    state = (await api.get(f'/api/possession/jobs/{job_id}')).json()
                    if state['state'] in ('ready', 'error'):
                        break
                    await asyncio.sleep(.1)
                self.assertEqual(state['state'], 'ready', state.get('error'))
                self.assertIsNone(state['manifest']['validation'])
                self.assertEqual(state['manifest']['frame_count'], 91)
                replay = (await api.get(f'/api/possession/jobs/{job_id}/frame', params={'time_s': .5})).json()
                self.assertEqual(replay['carrier'], 'home:10')
                self.assertEqual(replay['ball']['x'], 0.)
                self.assertEqual((await api.get(f'/api/possession/jobs/{job_id}/download/report.json')).status_code, 200)

    async def test_local_boundary_and_unknown_jobs_and_artifact_allowlist(self):
        parent = Path(__file__).resolve().parents[3] / 'tmp' / 'possession-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            path = Path(tmp)
            fixture(path / 'sample')
            store = JobStore(path / 'jobs', path / 'sample')
            app = create_app(store)
            transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
            async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1:8010') as api:
                self.assertEqual((await api.get('/api/possession/health')).status_code, 200)
                self.assertEqual((await api.get('/api/possession/jobs/missing')).status_code, 404)
                self.assertEqual((await api.get('/api/possession/health', headers={'Origin': 'https://attacker.example'})).status_code, 403)
                r = await api.post('/api/possession/jobs', json={'name': '../private.jsonl', 'source': 'raw'})
                self.assertEqual(r.status_code, 200)
                job_id = r.json()['id']
                self.assertTrue(store.current.folder.resolve().is_relative_to((path / 'jobs').resolve()))
                self.assertEqual((await api.get(f'/api/possession/jobs/{job_id}/download/private.key')).status_code, 404)
                self.assertEqual((await api.delete(f'/api/possession/jobs/{job_id}')).status_code, 200)
