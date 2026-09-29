"""One approval, one immutable execution snapshot; no ROS or flight IO."""
from dataclasses import dataclass
import math
import threading
import time
import uuid


class ApprovalContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PendingApproval:
    token: str
    proposal_id: str
    plan_json: str
    approved: bool
    epoch: int
    requested_at: float


class ApprovalExecutionLedger:
    def __init__(self, *, clock=time.monotonic, timeout_s=10.0, capacity=256):
        if not math.isfinite(timeout_s) or timeout_s <= 0 or capacity < 1:
            raise ValueError("invalid approval ledger limits")
        self.clock, self.timeout_s, self.capacity = clock, timeout_s, capacity
        self.lock = threading.RLock()
        self.connected = False
        self.epoch = 0
        self.latest = None
        self.proposals = {}
        self.pending = {}
        self.permits = {}
        # Bounded tombstones must not be evicted and make old IDs reusable.
        # Refuse new approvals at capacity until an explicit gateway restart.
        self.consumed = set()

    def invalidate(self, *, connected=None):
        with self.lock:
            self.epoch += 1
            self.pending.clear()
            self.permits.clear()
            if connected is not None:
                self.connected = connected

    def remember(self, proposal_id, plan_json, *, executable):
        with self.lock:
            value = (plan_json, bool(executable))
            if self.latest != proposal_id or self.proposals.get(proposal_id) != value:
                self.invalidate()
            self.latest = proposal_id
            self.proposals[proposal_id] = value
            while len(self.proposals) > self.capacity:
                del self.proposals[next(iter(self.proposals))]

    def begin(self, proposal_id, approved):
        with self.lock:
            if not self.connected:
                raise ApprovalContractError("gateway_disconnected", "approval requires a live gateway session")
            if proposal_id != self.latest or proposal_id not in self.proposals:
                raise ApprovalContractError("proposal_not_current", "approve the currently displayed proposal")
            if not approved:
                # Revocation supersedes an in-flight positive approval.
                self.pending.pop(proposal_id, None)
                self.permits = {k: v for k, v in self.permits.items() if k[0] != proposal_id}
                plan_json, _ = self.proposals[proposal_id]
                self.proposals[proposal_id] = (plan_json, False)
            if approved and proposal_id in self.consumed:
                raise ApprovalContractError("proposal_consumed", "this proposal already issued an execution attempt")
            if approved and len(self.consumed) >= self.capacity:
                raise ApprovalContractError("approval_capacity_reached", "gateway approval ledger is full; restart only when idle")
            if proposal_id in self.pending:
                raise ApprovalContractError("approval_in_progress", "approval is already in progress for this proposal")
            plan_json, executable = self.proposals[proposal_id]
            if approved and not executable:
                raise ApprovalContractError("proposal_not_executable", "only an OK flight proposal can execute")
            pending = PendingApproval(uuid.uuid4().hex, proposal_id, plan_json,
                                      bool(approved), self.epoch, self.clock())
            self.pending[proposal_id] = pending
            return pending

    def cancel(self, pending):
        with self.lock:
            if self.pending.get(pending.proposal_id) == pending:
                del self.pending[pending.proposal_id]

    def finish(self, pending, *, accepted, mission_id):
        with self.lock:
            if self.pending.get(pending.proposal_id) != pending:
                raise ApprovalContractError("approval_superseded", "approval belongs to an expired or replaced session")
            self.cancel(pending)
            elapsed = self.clock() - pending.requested_at
            if (not self.connected or pending.epoch != self.epoch
                    or not 0 <= elapsed <= self.timeout_s
                    or self.latest != pending.proposal_id
                    or self.proposals.get(pending.proposal_id, (None,))[0] != pending.plan_json):
                raise ApprovalContractError("approval_expired", "approval context changed or response arrived too late")
            if not accepted or not pending.approved:
                return None
            if not isinstance(mission_id, str) or not mission_id.strip():
                raise ApprovalContractError("approved_mission_missing", "accepted approval has no mission ID")
            source = dict(proposal_id=pending.proposal_id, mission_id=mission_id,
                          plan_json=pending.plan_json)
            self.permits[(pending.proposal_id, mission_id)] = (
                self.epoch, source, pending.requested_at, pending.requested_at+self.timeout_s)
            return dict(source)

    def claim_execution(self, source):
        with self.lock:
            key = (source["proposal_id"], source["mission_id"])
            permit = self.permits.pop(key, None)
            if (not self.connected or permit is None or permit[0] != self.epoch
                    or not permit[2] <= self.clock() <= permit[3]
                    or source["proposal_id"] in self.consumed
                    or source["plan_json"] != permit[1]["plan_json"]
                    or source.get("route_waypoints_enu")):
                raise ApprovalContractError("execution_not_authorized", "execution requires the exact unused approval snapshot")
            self.consumed.add(source["proposal_id"])
            return dict(permit[1])
