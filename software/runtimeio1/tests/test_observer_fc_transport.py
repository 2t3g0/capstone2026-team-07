import errno

import pytest

from jolgwa_uav.observer_fc_transport import (
    _allow_verified_pixhawk_acm_modem_control_epipe,
)


class FakePort:
    def __init__(self, dtr_error=None, rts_error=None):
        self.calls = []
        self.dtr_error = dtr_error
        self.rts_error = rts_error

    def _update_dtr_state(self):
        self.calls.append("dtr")
        if self.dtr_error:
            raise self.dtr_error

    def _update_rts_state(self):
        self.calls.append("rts")
        if self.rts_error:
            raise self.rts_error


def test_verified_pixhawk_ignores_only_modem_control_epipe():
    port = FakePort(
        BrokenPipeError(errno.EPIPE, "broken pipe"),
        BrokenPipeError(errno.EPIPE, "broken pipe"),
    )
    _allow_verified_pixhawk_acm_modem_control_epipe(port)
    port._update_dtr_state()
    port._update_rts_state()
    assert port.calls == ["dtr", "rts"]
    assert port.jolgwa_ignored_modem_control_epipe == [
        "_update_dtr_state", "_update_rts_state"]


@pytest.mark.parametrize("method", ["_update_dtr_state", "_update_rts_state"])
def test_other_modem_control_errors_remain_fatal(method):
    error = PermissionError(errno.EACCES, "permission denied")
    port = FakePort(error if method.endswith("dtr_state") else None,
                    error if method.endswith("rts_state") else None)
    _allow_verified_pixhawk_acm_modem_control_epipe(port)
    with pytest.raises(PermissionError):
        getattr(port, method)()


def test_normal_modem_control_operations_are_unchanged():
    port = FakePort()
    _allow_verified_pixhawk_acm_modem_control_epipe(port)
    port._update_dtr_state()
    port._update_rts_state()
    assert port.calls == ["dtr", "rts"]
    assert port.jolgwa_ignored_modem_control_epipe == []
