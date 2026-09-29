"""Startup-only mode selection. Observe is the safe default."""
import argparse
import rclpy
from .d435_observer_node import D435ObserverNode
from .jetson_d435i_bridge_node import JetsonD435iBridgeNode


def parse_mode(args=None):
    parser = argparse.ArgumentParser(description="D435 perception bridge (no FC command owner)")
    parser.add_argument("--mode", choices=("observe", "control"), default="observe",
                        help="observe: diagnostics only; control: existing SafetyDecision output")
    return parser.parse_known_args(args)


def main(args=None):
    options, ros_args = parse_mode(args)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = (D435ObserverNode() if options.mode == "observe" else
                JetsonD435iBridgeNode(default_jetson_url="http://127.0.0.1:8765"))
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
