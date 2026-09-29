import json
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import rclpy
from jolgwa_interfaces.msg import MissionProposal, TextCommand
from rclpy.node import Node

from .planner_client import PlannerClient, PlannerClientError
from .topic_names import MISSION_PROPOSAL, TEXT_COMMAND


STATUS_MAP = {
    "OK": MissionProposal.STATUS_OK,
    "NEED_CLARIFICATION": MissionProposal.STATUS_NEED_CLARIFICATION,
    "UNSUPPORTED": MissionProposal.STATUS_UNSUPPORTED,
}


class MissionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_planner")
        self.declare_parameter(
            "planner_endpoint", "http://127.0.0.1:9293/v1/plan/text"
        )
        self.declare_parameter("planner_timeout_s", 30.0)
        endpoint = self.get_parameter("planner_endpoint").value
        timeout_s = float(self.get_parameter("planner_timeout_s").value)
        self._client = PlannerClient(endpoint, timeout_s)
        self._publisher = self.create_publisher(
            MissionProposal, MISSION_PROPOSAL, 10
        )
        self.create_subscription(TextCommand, TEXT_COMMAND, self._on_command, 10)
        self._pending = deque()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active: tuple[TextCommand, Future] | None = None
        self.create_timer(0.05, self._poll)
        self.get_logger().info("mission planner bridge uses %s" % endpoint)

    def _on_command(self, message: TextCommand) -> None:
        self._pending.append(message)

    def _poll(self) -> None:
        if self._active is None and self._pending:
            command = self._pending.popleft()
            future = self._executor.submit(self._client.plan_text, command.text)
            self._active = (command, future)
            return
        if self._active is None or not self._active[1].done():
            return

        command, future = self._active
        self._active = None
        proposal = MissionProposal()
        proposal.stamp = self.get_clock().now().to_msg()
        proposal.proposal_id = str(uuid.uuid4())
        proposal.command_id = command.command_id
        proposal.raw_command = command.text
        try:
            plan = future.result()
            status_name = str(plan["status"])
            proposal.status = STATUS_MAP.get(
                status_name, MissionProposal.STATUS_ERROR
            )
            proposal.plan_json = json.dumps(
                plan, ensure_ascii=False, separators=(",", ":")
            )
            proposal.message = str(plan.get("message") or "")
            proposal.requires_approval = status_name == "OK"
        except (PlannerClientError, KeyError, TypeError, ValueError) as exc:
            proposal.status = MissionProposal.STATUS_ERROR
            proposal.message = str(exc)
            proposal.requires_approval = False
            self.get_logger().error("mission planning failed: %s" % exc)
        self._publisher.publish(proposal)
        self.get_logger().info(
            "published proposal %s for command %s with status %s"
            % (proposal.proposal_id, command.command_id, proposal.status)
        )

    def destroy_node(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
