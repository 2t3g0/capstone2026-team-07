"""Fixed, opt-in owned-SITL preparation after an accepted dashboard action.

No ROS approval or navigation publisher exists here. The caller must already
own the accepted mission; the child independently observes its actual approval.
"""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


def validate_profile(*, enabled, simulation_only, allow_real_hardware, use_sim_time,
                     diagnostic_relaxed_timing, capture_then_home):
    if type(enabled) is not bool:
        raise ValueError('owned SIM preparation enable must be boolean')
    if enabled and not (simulation_only is True and allow_real_hardware is False
            and use_sim_time is True and diagnostic_relaxed_timing is True
            and capture_then_home is True):
        raise ValueError('owned dashboard preparation requires the explicit SIM-only incident profile')


class OwnedSimPreparation:
    def __init__(self, configuration, *, popen=subprocess.Popen):
        self.configuration = Path(configuration).resolve(strict=True)
        self.root = Path(__file__).resolve().parents[4]
        if not self.configuration.is_relative_to(self.root / 'artifacts'):
            raise ValueError('Preparation configuration must be a project artifact')
        self.config = json.loads(self.configuration.read_text())
        if self.config.get('schema') != 1 or self.config.get('simulation_only') is not True:
            raise ValueError('Not an owned dashboard SIM configuration')
        self._popen = popen
        self.process = None
        self.directory = None
        self.evidence_directory = None
        self.mission_id = self.proposal_id = None
        self.handed_off = False
        self._log = None

    def start(self, mission_id, proposal_id):
        if self.process is not None:
            raise RuntimeError('This preparation session is single-use')
        # IDs originate in the real accepted action, not a caller-selected path.
        mission_id, proposal_id = str(uuid.UUID(mission_id)), str(uuid.UUID(proposal_id))
        self.mission_id, self.proposal_id = mission_id, proposal_id
        self.evidence_directory = self.configuration.parent / ('mission_' + mission_id)
        self.evidence_directory.mkdir(exist_ok=False)
        # Never use DrvFS/NTFS artifacts as the command/status transport. This
        # private Linux directory is generated here, not read from configuration.
        self.directory = Path(tempfile.mkdtemp(prefix='jolgwa-dashboard-' + mission_id + '-', dir='/tmp'))
        self._log = (self.directory / 'worker.log').open('xb')
        self.process = self._popen([sys.executable,
            str(self.root / 'scripts/owned_dashboard_preparation_worker.py'),
            '--configuration', str(self.configuration), '--mission-id', mission_id,
            '--proposal-id', proposal_id, '--manager-pid', str(os.getpid()),
            '--runtime-dir', str(self.directory),
            '--manager-start-ticks', Path('/proc/self/stat').read_text().split()[21]],
            cwd=self.root, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)

    def status(self):
        if self.process is None:
            return {'stage': 'not_started', 'error': ''}
        try:
            status = json.loads((self.directory / 'state.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            status = {'stage': 'starting', 'error': ''}
        if status.get('mission_id', self.mission_id) != self.mission_id:
            raise RuntimeError('Preparation mission identity changed')
        if status.get('proposal_id', self.proposal_id) != self.proposal_id:
            raise RuntimeError('Preparation proposal identity changed')
        if self.process.poll() is not None and status.get('stage') != 'closed':
            status['error'] = status.get('error') or 'preparation worker exited before normal cleanup'
        return status

    def handoff(self):
        status = self.status()
        if status.get('stage') != 'perception_ready' or status.get('error'):
            raise RuntimeError('Actual preparation is not ready for route handoff')
        with (self.directory / 'handoff.json').open('x') as stream:
            json.dump({'mission_id': self.mission_id, 'proposal_id': self.proposal_id,
                       'monotonic_s': time.monotonic()}, stream)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.status()
            if status.get('error'):
                raise RuntimeError(status['error'])
            if status.get('stage') == 'route_handoff_heartbeat_only':
                self.handed_off = True
                return
            time.sleep(.02)
        raise TimeoutError('worker did not acknowledge route handoff')

    def close(self):
        if self.process is None:
            if self._log is not None:
                self._log.close()
            return {'stage': 'not_started', 'error': ''}
        if self.process.poll() is None:
            # Child signal handler requests the unchanged helper's native RTL.
            # Never kill a broad process group or another mission's helper.
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=195)
            except subprocess.TimeoutExpired:
                return {'stage': 'cleanup_unconfirmed', 'error': 'owned native landing cleanup still running'}
        if self._log is not None:
            self._log.close()
        evidence = getattr(self, 'evidence_directory', None)
        if evidence is not None and not (evidence / 'worker.log').exists():
            try:
                with (self.directory / 'worker.log').open('rb') as source, (evidence / 'worker.log').open('xb') as target:
                    shutil.copyfileobj(source, target)
            except OSError:
                pass  # Closed artifact export is not command/readiness evidence.
        return self.status()
