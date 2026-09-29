"""Replay the measured Home-only delta; keep the 2m limit unchanged."""
import json
from pathlib import Path
import pytest
from test_arm_home_order import BASE, locked_state, heartbeat


def test_ulog67_large_home_metadata_change_remains_rejected():
    evidence=json.loads((Path(__file__).parent/'fixtures/home_flight_67.json').read_text())
    before,after=evidence['home_transitions'][1:3]
    delta=after['alt']-before['alt']
    assert delta==pytest.approx(5.3590850830078125)
    assert (after['z']-before['z'])==pytest.approx(-delta)
    state,_=locked_state()
    heartbeat(state,BASE+20_000_000)
    frozen=dict(state.home_position)
    home=dict(latitude=352350126,longitude=1290748631,
        altitude=round((42.8+delta)*1000),x=1.,y=2.,z=-1.-delta)
    state.accept('HOME_POSITION',home,1,1,BASE+40_000_000)
    assert not state.home_correction_valid
    assert state.home_correction_detail=='home_correction_magnitude_exceeded'
    for i in (1,2):
        state.accept('HOME_POSITION',home,1,1,BASE+40_000_000+i)
    assert state.home_epoch_failure_latched
    assert state.home_position==frozen
