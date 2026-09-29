"""Verify a source or installed release against ROUTE_RELEASE_MANIFEST.json."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest_path = root/"ROUTE_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    gateway = root/'ros2_ws/src/jolgwa_ros/jolgwa_ros/operator_gateway_node.py'
    tree = ast.parse(gateway.read_text(encoding='utf-8'))
    assignments = {target.id: node.value.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets if isinstance(target, ast.Name)}
    for name, key in [('BUILD_ID', 'build_id'), ('PROTOCOL_VERSION', 'protocol_version')]:
        if assignments.get(name) != manifest.get(key):
            failures.append('gateway/manifest ' + key)
    advertised = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                    advertised[key.value] = value.value
    for key in ('flight_output_handshake_version', 'altitude_reference_version',
                'home_correction_version', 'status_freshness_version',
                'emergency_control_version', 'low_speed_profile_version'):
        if advertised.get(key) != manifest.get(key):
            failures.append('capability/manifest ' + key)
    for relative, expected in manifest["files_sha256"].items():
        path = root/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        print(("OK" if actual == expected else "BAD"), actual, relative)
        if actual != expected:
            failures.append(relative)
    if failures:
        raise SystemExit("release verification failed: " + ", ".join(failures))
    print("RELEASE_OK", manifest["build_id"], "protocol", manifest["protocol_version"])


if __name__ == "__main__":
    main()
