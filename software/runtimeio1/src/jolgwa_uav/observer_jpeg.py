"""Validate real RGB pixels, not merely receipt of a CompressedImage header."""
from __future__ import annotations

MAX_JPEG_BYTES = 4_000_000


def validate_observer_jpeg(data, image_format: str) -> tuple[int, int]:
    """Return (width, height) only for a bounded, decodable 8-bit RGB JPEG.

    Inspect SOF dimensions before asking OpenCV to allocate a decoded image.
    This is a payload check, not a proof of scene quality or sensor freshness.
    """
    if not isinstance(image_format, str) or not any(s in image_format.lower() for s in ('jpeg', 'jpg')):
        raise ValueError('camera_frame_not_jpeg')
    if not 4 <= len(data) <= MAX_JPEG_BYTES:
        raise ValueError('rgb_jpeg_empty_or_oversized')
    payload = bytes(data)
    if payload[:2] != b'\xff\xd8' or payload[-2:] != b'\xff\xd9':
        raise ValueError('rgb_jpeg_markers_invalid')
    offset, dimensions = 2, None
    while offset < len(payload) - 2:
        if payload[offset] != 0xff:
            raise ValueError('rgb_jpeg_segment_invalid')
        while offset < len(payload) and payload[offset] == 0xff:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker in (0xda, 0xd9):
            break
        if marker == 0x01 or 0xd0 <= marker <= 0xd7:
            continue
        if offset + 2 > len(payload):
            raise ValueError('rgb_jpeg_segment_truncated')
        length = int.from_bytes(payload[offset:offset+2], 'big')
        if length < 2 or offset + length > len(payload):
            raise ValueError('rgb_jpeg_segment_truncated')
        if marker in (0xc0, 0xc1, 0xc2):
            if length < 8:
                raise ValueError('rgb_jpeg_sof_invalid')
            height = int.from_bytes(payload[offset+3:offset+5], 'big')
            width = int.from_bytes(payload[offset+5:offset+7], 'big')
            if payload[offset+2] != 8 or payload[offset+7] != 3 or not (2 <= width <= 1920 and 2 <= height <= 1080):
                raise ValueError('rgb_jpeg_dimensions_or_channels_invalid')
            dimensions = (width, height)
            break
        offset += length
    if dimensions is None:
        raise ValueError('rgb_jpeg_sof_missing')
    import cv2
    import numpy as np
    try:
        pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError('rgb_jpeg_decode_failed') from exc
    width, height = dimensions
    if pixels is None or pixels.shape != (height, width, 3):
        raise ValueError('rgb_jpeg_decode_failed')
    return dimensions
