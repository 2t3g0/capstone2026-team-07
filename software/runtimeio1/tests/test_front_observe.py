"""Run against the deployed ROS dependencies without ROS publishers or a camera."""
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from urllib.request import urlopen
import jolgwa_ros

from jolgwa_ros import scenario_camera_status as status
from jolgwa_ros.scenario_observe_view import CameraObservation, handler_with_observation


class ObserveTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.observation = CameraObservation(clock=lambda: self.now)
        self.store = self.observation.store
        self.store.depth_at = self.store.rgb_at = self.now
        self.store.depth_stamp = 100
        self.store.stats = dict(front_near_m=2., front_median_m=2.5, upper_roi_near_m=4.,
            front_obstacle_fraction=.2, center_valid_fraction=.8,
            upper_obstacle_fraction=.1, upper_valid_fraction=.8, lower_valid_fraction=.8,
            perception_thresholds=dict(trigger_distance_m=3., release_distance_m=5.5,
                min_obstacle_fraction=.02, min_valid_fraction=.25))

    def test_current_blocked_does_not_require_pose(self):
        value = self.observation.snapshot('abc')
        self.assertEqual(value['perception']['assessment'], 'BLOCKED')
        self.assertEqual(value['report']['assessment'], 'UNKNOWN')
        self.assertEqual(value['request_nonce'], 'abc')
        self.assertFalse(value['flight_commands_enabled'])

    def test_clear(self):
        self.store.stats.update(front_near_m=6., front_obstacle_fraction=0.)
        self.assertEqual(self.observation.snapshot()['perception']['assessment'], 'CLEAR')

    def test_climb_candidate(self):
        self.store.stats.update(upper_roi_near_m=6., upper_obstacle_fraction=0.)
        value = self.observation.snapshot()['perception']
        self.assertEqual(value['assessment'], 'CLIMB_REQUIRED')
        self.assertFalse(value['climb_path_verified'])

    def test_bad_center_quality(self):
        self.store.stats['center_valid_fraction'] = .1
        self.assertEqual(self.observation.snapshot()['perception']['assessment'], 'UNKNOWN')

    def test_missing_upper_keeps_confirmed_front_obstacle(self):
        self.store.stats.update(upper_roi_near_m=None, upper_valid_fraction=0.)
        self.assertEqual(self.observation.snapshot()['perception']['assessment'], 'BLOCKED')

    def test_stream_expiration(self):
        self.now += .5
        value = self.observation.snapshot()
        self.assertEqual(value['perception']['assessment'], 'UNKNOWN')
        self.assertEqual(value['display_valid_for_s'], 0.)
        self.assertEqual(value['sensor'], {})

    def test_rgb_expiration(self):
        self.store.rgb_at -= .6
        self.assertEqual(self.observation.snapshot()['perception']['assessment'], 'UNKNOWN')

    def test_delayed_source_never_gets_new_lease(self):
        message = NS(header=NS(stamp=NS(sec=1, nanosec=0)))
        with patch.object(status, 'depth_statistics', return_value=self.store.stats):
            self.observation.depth(message, .4)
        self.assertAlmostEqual(self.observation.snapshot()['display_valid_for_s'], .1)
        self.now += .11
        self.assertEqual(self.observation.snapshot()['perception']['assessment'], 'UNKNOWN')

    def test_repeated_depth_timestamp_invalidates(self):
        message = NS(header=NS(stamp=NS(sec=0, nanosec=100)))
        self.observation.depth(message, 0.)
        self.assertEqual(self.observation.snapshot()['perception']['assessment'], 'UNKNOWN')

    def test_bad_source_invalidates_without_native_call(self):
        for age in (float('nan'), -.1, .5):
            with patch.object(status, 'depth_statistics') as calculate:
                self.observation.depth(NS(), age)
                calculate.assert_not_called()
            self.assertEqual(self.observation.snapshot()['perception']['assessment'], 'UNKNOWN')

    def test_policy_ages_are_added_and_expire(self):
        self.observation.safety(NS(observation_age_s=.2, state=2, reason='front', source='depth'), .2)
        self.assertEqual(self.observation.snapshot()['movement_policy']['state'], 'HOLD')
        self.now += .11
        self.assertEqual(self.observation.snapshot()['movement_policy']['state'], 'UNKNOWN')

    def test_native_depth_on_real_library(self):
        import numpy as np
        # Actual C++ ROI path, synthetic pixels only within this unit test.
        data = np.full((480, 640), 2000, dtype='<u2').tobytes()
        sensor = status.depth_statistics(NS(encoding='16UC1', height=480, width=640,
            step=1280, data=data, is_bigendian=False))
        self.assertAlmostEqual(sensor['front_near_m'], 2., places=3)
        self.assertGreater(sensor['center_valid_fraction'], .99)

    def test_launch_lifetime_guard_keeps_current_parent(self):
        from jolgwa_ros.scenario_incident_node import ScenarioIncidentNode
        stat = Path('/proc', str(os.getpid()), 'stat').read_text()
        owner = NS(launch_parent=str(os.getpid()),
                   launch_start=stat.rsplit(')', 1)[1].split()[19])
        with patch('jolgwa_ros.scenario_incident_node.rclpy.shutdown') as shutdown:
            ScenarioIncidentNode.check_launch_parent(owner)
            shutdown.assert_not_called()

    def test_launch_lifetime_guard_stops_after_owner_exit(self):
        from jolgwa_ros.scenario_incident_node import ScenarioIncidentNode
        owner = NS(launch_parent='missing-owner', launch_start='0')
        with patch('jolgwa_ros.scenario_incident_node.rclpy.ok', return_value=True), \
             patch('jolgwa_ros.scenario_incident_node.rclpy.shutdown') as shutdown:
            ScenarioIncidentNode.check_launch_parent(owner)
            shutdown.assert_called_once()

    def test_http_page_status_and_legacy_paths(self):
        class Base(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(204)
                self.end_headers()
            def log_message(self, *args): pass
        bench = NS(monitor_snapshot=lambda nonce: {'request_nonce': nonce, 'status': 'RUNNING'})
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_with_observation(Base, self.observation, bench))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with urlopen(origin + '/') as response:
                self.assertIn('현재 카메라 장애물 판정', response.read().decode())
            with urlopen(origin + '/observe/v1/status?nonce=test') as response:
                self.assertEqual(json.load(response)['request_nonce'], 'test')
            with urlopen(origin + '/monitor/v1/status?nonce=test') as response:
                value = json.load(response)
                self.assertEqual(value['status'], 'RUNNING')
                self.assertEqual(value['camera_observation']['perception']['assessment'], 'BLOCKED')
            with urlopen(origin + '/preview.jpg') as response:
                self.assertEqual(response.status, 204)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
