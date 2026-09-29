import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .topic_names import TEXT_INPUT


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Publish one patrol command")
    parser.add_argument("text")
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = Node("send_text_command")
    publisher = node.create_publisher(String, TEXT_INPUT, 10)
    deadline = node.get_clock().now().nanoseconds + 2_000_000_000
    while publisher.get_subscription_count() == 0:
        if node.get_clock().now().nanoseconds >= deadline:
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    message = String()
    message.data = parsed.text
    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.25)
    node.destroy_node()
    rclpy.shutdown()
