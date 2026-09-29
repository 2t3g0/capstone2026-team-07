from jolgwa_ros.low_speed import (
    LOW_SPEED_1M_V1,
    LOW_SPEED_2M_V1,
    capture_then_home_for_profile,
)


def test_capture_then_home_is_safe_before_active_profile_initialization():
    assert capture_then_home_for_profile(
        event_completion_policy="capture_then_rtl"
    ) is True


def test_low_speed_profile_never_uses_capture_then_home():
    assert capture_then_home_for_profile(
        LOW_SPEED_2M_V1, "capture_then_rtl"
    ) is False
    assert capture_then_home_for_profile(
        LOW_SPEED_1M_V1, "capture_then_rtl"
    ) is False
