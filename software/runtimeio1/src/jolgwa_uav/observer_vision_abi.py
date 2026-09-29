"""Fail before opening a camera if the dedicated vision overlay is inconsistent."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess


def check_dependencies(output: str) -> None:
    if 'not found' in output:
        raise ValueError('observer_vision_dependency_missing')
    libraries = re.findall(r'libopencv_\w+\.so\.[\w.]+', output)
    if not libraries or any(not name.endswith(('.so.408', '.so.4.8', '.so.4.8.0')) for name in libraries):
        raise ValueError('observer_vision_opencv_abi_mismatch')


def main():
    root = Path('/home/jetson/jolgwa/observer_vision_ws/install')
    targets = (
        root / 'cv_bridge/lib/libcv_bridge.so',
        root / 'compressed_image_transport/lib/libcompressed_image_transport.so',
        root / 'realsense2_camera/lib/librealsense2_camera.so',
    )
    for target in targets:
        if not target.is_file():
            raise RuntimeError(f'observer_vision_overlay_missing: {target}')
        output = subprocess.check_output(['ldd', str(target)], text=True)
        check_dependencies(output)
        if target.name != 'libcv_bridge.so' and str(root / 'cv_bridge/lib/libcv_bridge.so') not in output:
            raise RuntimeError('observer_vision_cv_bridge_not_from_private_overlay')
    print('Observer camera ABI preflight: OpenCV 4.8 only, private cv_bridge/transport.', flush=True)


if __name__ == '__main__':
    main()
