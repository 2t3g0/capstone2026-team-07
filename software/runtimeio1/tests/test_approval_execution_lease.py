import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/jolgwa_ros"))

from jolgwa_ros.approval_execution import ApprovalContractError, ApprovalExecutionLedger


def test_one_approval_yields_exactly_one_execution_claim():
    clock = [10.0]
    ledger = ApprovalExecutionLedger(clock=lambda: clock[0], timeout_s=5.0)
    ledger.invalidate(connected=True)
    ledger.remember("proposal", '{"status":"OK"}', executable=True)
    pending = ledger.begin("proposal", True)
    execution = ledger.finish(pending, accepted=True, mission_id="mission")
    assert ledger.claim_execution(execution) == execution
    with pytest.raises(ApprovalContractError, match="exact unused approval"):
        ledger.claim_execution(execution)


def test_execution_permit_expires_at_five_seconds():
    clock = [10.0]
    ledger = ApprovalExecutionLedger(clock=lambda: clock[0], timeout_s=5.0)
    ledger.invalidate(connected=True)
    ledger.remember("proposal", "{}", executable=True)
    pending = ledger.begin("proposal", True)
    execution = ledger.finish(pending, accepted=True, mission_id="mission")
    clock[0] = 15.01
    with pytest.raises(ApprovalContractError, match="exact unused approval"):
        ledger.claim_execution(execution)
