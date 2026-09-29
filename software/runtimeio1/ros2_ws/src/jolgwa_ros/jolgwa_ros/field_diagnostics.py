"""Sampled metadata diagnostics; never supplies or refreshes flight evidence."""
import json
import math
import os
import time


def record(node, stage, **values):
    if not (getattr(node, '_scenario_enabled', False)
            or os.environ.get('JOLGWA_SCENARIO_PROFILE') == 'FRONT_DETECT_2M_PASS_3M_V1'):
        return
    now = time.monotonic()
    records = getattr(node, '_field_diagnostics', None)
    if records is None:
        records = node._field_diagnostics = {}
    previous = records.get(stage, (-math.inf, 0, None))
    count = previous[1]+1
    last_good = now if values.get('valid') is True else previous[2]
    records[stage] = (previous[0], count, last_good)
    if now-previous[0] < 1.:
        return
    records[stage] = (now, 0, last_good)
    emit = getattr(node.get_logger(), 'info', None) if hasattr(node, 'get_logger') else None
    if emit is not None:
        emit(json.dumps(dict(event='field_input_diagnostic', stage=stage,
            monotonic_s=now, samples=count, last_good_s=last_good, **values), default=str))
