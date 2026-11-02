"""Local-only upload, cancellable research worker and compact replay API."""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from local_boundary import LocalAnalyticsBoundary

ROOT = Path(__file__).resolve().parents[2]
DOWNLOADS = {'report.json', 'REPORT.md', 'frames.csv', 'predicted_events.csv', 'event_matches.csv',
             'actor_validation.csv', 'players.csv', 'control_annotations.csv'}


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def finite(value):
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None


class DemoResult:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.report = json.loads((self.folder / 'report.json').read_text(encoding='utf-8'))
        self.rows = read_csv(self.folder / 'frames.csv')
        self.events = read_csv(self.folder / 'predicted_events.csv')
        for i, e in enumerate(self.events):
            e.update(id=i, period=int(e['period']), time_s=float(e['time_s']), end_time_s=float(e['end_time_s']))
        self.names = {r['player']: r.get('name') or r['player'] for r in read_csv(self.folder / 'players.csv')}
        with np.load(self.folder / 'trajectories.npz', allow_pickle=False) as z:
            for key in ['time', 'period', 'frame', 'xy', 'ball', 'height', 'player_velocity', 'ball_velocity']:
                setattr(self, key, z[key])
            self.keys = list(z['keys'])
        if not len(self.time) or len(self.rows) != len(self.time):
            raise ValueError('Incomplete replay outputs')
        for key in self.keys:
            self.names.setdefault(key, key)
        self.players = []
        for key in self.keys:
            self.players.append(dict(key=key, name=self.names[key], side=key.split(':')[0], jersey=key.split(':')[1],
                                     interaction_seconds=self.report['summary'].get('player_interaction_seconds', {}).get(key, 0.),
                                     touches=sum(e['event_type'] == 'touch_candidate' and e['actor'] == key for e in self.events),
                                     passes=sum(e['event_type'] == 'completed_pass_candidate' and e['actor'] == key for e in self.events)))

    def manifest(self):
        cuts = np.r_[0, np.flatnonzero((np.diff(self.period) != 0) | (np.diff(self.time) > .2)) + 1, len(self.time)]
        segments = [dict(period=int(self.period[a]), start=float(self.time[a]), end=float(self.time[b - 1]))
                    for a, b in zip(cuts, cuts[1:])]
        validation = self.report['validation']
        available = validation.get('available', any(v.get('references', 0) for v in validation.get('event_detection', {}).values()))
        return dict(source=self.report['source'], frame_count=len(self.time), start=float(self.time[0]),
                    end=float(self.time[-1]), segments=segments, players=self.players,
                    candidate_counts=self.report['candidate_counts'], event_count=len(self.events),
                    summary=self.report['summary'], validation=validation if available else None,
                    control_classifier=self.report.get('control_classifier', 'untrained'),
                    parameters=self.report.get('parameters', {}), limitations=self.report.get('limitations', []))

    def frame_at(self, time_s):
        i = min(int(np.searchsorted(self.time, time_s)), len(self.time) - 1)
        if i and abs(self.time[i - 1] - time_s) < abs(self.time[i] - time_s):
            i -= 1
        row = self.rows[i]
        ball = self.ball[i]
        players = []
        for slot, key in enumerate(self.keys):
            x, y = self.xy[i, slot]
            if not np.isfinite([x, y]).all():
                continue
            players.append(dict(key=key, name=self.names[key], side=key.split(':')[0], jersey=key.split(':')[1],
                                x=finite(x), y=finite(y), speed=finite(np.linalg.norm(self.player_velocity[i, slot])),
                                ball_distance=finite(np.linalg.norm(self.xy[i, slot] - ball))))
        now = float(self.time[i])
        near = [e for e in self.events if e['time_s'] - .6 <= now <= e['end_time_s'] + .6]
        near.sort(key=lambda e: abs(e['time_s'] - now))
        return dict(time_s=now, frame_num=int(self.frame[i]), period=int(self.period[i]), players=players,
                    ball=dict(x=finite(ball[0]), y=finite(ball[1]), z=finite(self.height[i]),
                              speed=finite(np.linalg.norm(self.ball_velocity[i])),
                              reconstructed=row.get('reconstructed_ball') == 'True') if np.isfinite(ball).all() else None,
                    carrier=row.get('carrier', ''), team=row.get('team', ''), status=row['status'],
                    control_status=row.get('control_status', 'unknown'), previous_owner=row.get('previous_owner', ''),
                    tracking_weight=finite(row.get('tracking_quality', 0)), nearby_events=near[:6])

    def events_page(self, kind, player, page, size):
        needle = player.casefold()
        selected = [e for e in self.events if (not kind or e['event_type'] == kind) and
                    (not needle or any(needle in (self.names.get(e.get(k), e.get(k, '')) + ' ' + e.get(k, '')).casefold()
                                       for k in ['actor', 'target', 'challenger']))]
        return dict(total=len(selected), page=page, page_size=size, events=selected[page * size:(page + 1) * size])


@dataclass
class Job:
    id: str
    name: str
    source: str
    folder: Path
    state: str = 'uploading'
    stage: str = 'Waiting for selected files'
    progress: float = 0.
    error: str | None = None
    process: subprocess.Popen | None = None
    result: DemoResult | None = None


class JobStore:
    def __init__(self, root=None, sample=None):
        self.root = Path(root or ROOT / 'output' / 'possession-demo' / 'jobs')
        self.sample = Path(sample or ROOT / 'output' / 'research-possession-3816-smoothed')
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = None
        self.lock = threading.RLock()

    def reserve(self, name, source):
        with self.lock:
            if self.current and self.current.state in ('uploading', 'analyzing', 'loading'):
                raise HTTPException(409, 'An analysis is already running. Resume it or cancel it first.')
            old = self.current
            if old:
                old.result = None
                target = old.folder.resolve()
                if target.parent == self.root.resolve() and target.is_relative_to(self.root.resolve()):
                    shutil.rmtree(target, ignore_errors=True)  # only the verified demo-owned job directory
            job_id = uuid4().hex
            folder = self.root / job_id
            folder.mkdir()
            self.current = Job(job_id, name, source, folder)
            return self.current

    def get(self, job_id):
        with self.lock:
            if not self.current or self.current.id != job_id:
                raise HTTPException(404, 'Analysis not found. Load a file or the analysed demo match.')
            return self.current

    def status(self, job):
        response = dict(id=job.id, name=job.name, source=job.source, state=job.state,
                        stage=job.stage, progress=job.progress, error=job.error)
        if job.result and job.state == 'ready':
            response['manifest'] = job.result.manifest()
        return response

    def cancel(self, job_id):
        job = self.get(job_id)
        with self.lock:
            job.state, job.stage = 'cancelled', 'Analysis cancelled'
            if job.process and job.process.poll() is None:
                job.process.terminate()
            job.result = None
        return self.status(job)

    def start(self, job, has_events):
        with self.lock:
            if job.state != 'uploading':
                raise HTTPException(409, 'This upload was cancelled or has already started.')
            job.state, job.stage, job.progress = 'analyzing', 'Reading tracking file', .08
            command = [sys.executable, '-B', str(ROOT / 'scripts' / 'research_possession.py'),
                       '--tracking', str(job.folder / 'tracking.jsonl'), '--source', job.source,
                       '--output', str(job.folder / 'result')]
            if has_events:
                command += ['--events', str(job.folder / 'events.json')]
            job.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace',
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        threading.Thread(target=self._monitor, args=(job,), daemon=True).start()

    def _monitor(self, job):
        stages = {'distinct frames': ('Preparing trajectories', .2), 'player trajectories': ('Smoothing player movement', .35),
                  'ball trajectory': ('Smoothing ball movement', .5), 'Inferring': ('Detecting interactions and events', .65),
                  'Predictions frozen': ('Validating and preparing replay', .82), 'Wrote': ('Loading replay', .94)}
        timer = threading.Timer(600, lambda: self.cancel(job.id) if self.current is job else None)
        timer.daemon = True
        timer.start()
        tail = []
        try:
            with job.process.stdout as output:
                for line in output:
                    tail = (tail + [line.strip()])[-12:]
                    for key, (label, fraction) in stages.items():
                        if key in line and job.state != 'cancelled':
                            job.stage, job.progress = label, fraction
            code = job.process.wait()
            if job.state == 'cancelled':
                return
            if code:
                raise ValueError(next((s for s in reversed(tail) if s), 'The analysis worker stopped.'))
            job.state, job.stage = 'loading', 'Loading replay'
            result = DemoResult(job.folder / 'result')
            if job.state != 'cancelled':
                job.result, job.state, job.stage, job.progress = result, 'ready', 'Ready', 1.
        except Exception as exc:
            if job.state != 'cancelled':
                job.state, job.error = 'error', str(exc)
        finally:
            timer.cancel()

    def load_sample(self):
        if not (self.sample / 'trajectories.npz').exists():
            raise HTTPException(404, 'No analysed sample is available. Choose your tracking file instead.')
        job = self.reserve('Match 3816 · analysed recording', 'smoothed')
        job.state, job.stage, job.progress = 'loading', 'Opening analysed recording', .5

        def load():
            try:
                result = DemoResult(self.sample)
                if job.state != 'cancelled':
                    job.result, job.state, job.stage, job.progress = result, 'ready', 'Ready', 1.
            except Exception as exc:
                job.state, job.error = 'error', str(exc)
        threading.Thread(target=load, daemon=True).start()
        return self.status(job)


class NewJob(BaseModel):
    name: str = Field(min_length=1, max_length=240)
    source: Literal['raw', 'smoothed'] = 'smoothed'


def create_app(store=None):
    store = store or JobStore()

    @asynccontextmanager
    async def lifespan(_app):
        spool = store.root / 'multipart'
        spool.mkdir(exist_ok=True)
        tempfile.tempdir = str(spool)
        yield
        if store.current:
            store.cancel(store.current.id)

    app = FastAPI(title='Local possession demo', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origin_regex=r'^http://(localhost|127\.0\.0\.1)(:\d+)?$',
                       allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['*'])
    app.add_middleware(LocalAnalyticsBoundary)
    prefix = '/api/possession'

    @app.get(prefix + '/health')
    def health():
        return dict(status='ok', sample_available=(store.sample / 'trajectories.npz').exists(),
                    current_job=store.status(store.current) if store.current else None)

    @app.post(prefix + '/jobs')
    def reserve(payload: NewJob):
        return store.status(store.reserve(payload.name, payload.source))

    @app.post(prefix + '/sample')
    def sample():
        return store.load_sample()

    @app.get(prefix + '/jobs/{job_id}')
    def status(job_id: str):
        return store.status(store.get(job_id))

    @app.delete(prefix + '/jobs/{job_id}')
    def cancel(job_id: str):
        return store.cancel(job_id)

    @app.post(prefix + '/jobs/{job_id}/files')
    async def upload(job_id: str, tracking: UploadFile = File(...), events: UploadFile | None = File(None)):
        job = store.get(job_id)
        if job.state != 'uploading':
            raise HTTPException(409, 'Upload is no longer active.')
        if not (tracking.filename or '').lower().endswith('.jsonl'):
            job.state = 'error'
            raise HTTPException(422, 'Choose a tracking .jsonl file.')
        try:
            for incoming, filename, maximum in [(tracking, 'tracking.jsonl', 1_073_000_000), (events, 'events.json', 64 * 1024 * 1024)]:
                if incoming is None:
                    continue
                written = 0
                with (job.folder / filename).open('wb') as output:
                    while chunk := await incoming.read(1024 * 1024):
                        if job.state == 'cancelled':
                            raise HTTPException(409, 'Upload cancelled.')
                        written += len(chunk)
                        if written > maximum:
                            raise HTTPException(413, 'Selected file exceeds the local upload limit.')
                        output.write(chunk)
                if not written:
                    raise HTTPException(422, 'Selected files must not be empty.')
            store.start(job, events is not None)
        except BaseException:
            if job.state != 'cancelled':
                job.state = 'error'
            raise
        finally:
            await tracking.close()
            if events is not None:
                await events.close()
        return store.status(job)

    def ready(job_id):
        job = store.get(job_id)
        if job.state != 'ready' or not job.result:
            raise HTTPException(409, 'Analysis is not ready.')
        return job.result

    @app.get(prefix + '/jobs/{job_id}/frame')
    def frame(job_id: str, time_s: float):
        if not math.isfinite(time_s):
            raise HTTPException(422, 'Playback time must be finite.')
        return ready(job_id).frame_at(time_s)

    @app.get(prefix + '/jobs/{job_id}/events')
    def events_page(job_id: str, kind: str = '', player: str = '', page: int = Query(0, ge=0), size: int = Query(40, ge=1, le=200)):
        return ready(job_id).events_page(kind, player, page, size)

    @app.get(prefix + '/jobs/{job_id}/download/{filename}')
    def download(job_id: str, filename: str):
        if filename not in DOWNLOADS:
            raise HTTPException(404, 'Artifact not available.')
        path = ready(job_id).folder / filename
        if not path.is_file():
            raise HTTPException(404, 'Artifact not available.')
        return FileResponse(path, filename=filename)

    return app


app = create_app()
