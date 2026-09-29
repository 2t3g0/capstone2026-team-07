import uuid

import rclpy
from jolgwa_interfaces.msg import TextCommand
from rclpy.node import Node
from std_msgs.msg import String

from .topic_names import TEXT_COMMAND, TEXT_INPUT


class CommandInputNode(Node):
    def __init__(self) -> None:
        super().__init__("command_input")
        self._publisher = self.create_publisher(TextCommand, TEXT_COMMAND, 10)
        self.create_subscription(String, TEXT_INPUT, self._on_text, 10)
        self.get_logger().info("text command input is ready on %s" % TEXT_INPUT)

    def _on_text(self, source: String) -> None:
        text = source.data.strip()
        if not text:
            self.get_logger().warning("ignored an empty text command")
            return
        message = TextCommand()
        message.stamp = self.get_clock().now().to_msg()
        message.command_id = str(uuid.uuid4())
        message.source = TextCommand.SOURCE_TEXT
        message.text = text
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommandInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
