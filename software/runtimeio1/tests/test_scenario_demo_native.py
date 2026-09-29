import numpy as np
import pytest
from jolgwa_uav.vertical_avoidance import VerticalObstacleAvoidanceCore, VerticalAvoidanceConfig

def test_native_demo_new_obstacle_cannot_reuse_previous_extent_or_altitude(monkeypatch):
    import jolgwa_uav.jetson_compute_service as service
    runtime=service.JetsonNativeDepthRuntime(VerticalObstacleAvoidanceCore(
        VerticalAvoidanceConfig(require_geometry=True,scenario_demo=True,
            trigger_samples=3,emergency_margin_m=0.,release_samples=1)))
    geometry=dict(geometry_valid=True,obstacle_extent_valid=True,
        obstacle_far_north_m=99.,obstacle_far_east_m=0.,roof_clearance_verified=False)
    monkeypatch.setattr(service,'_observe_native_geometry',lambda *a,**kw:(object(),geometry.copy()))
    runtime._nominal_avoidance_altitude_m=30.
    wall=np.full((48,64),8.,dtype=np.float32)
    wall[19:34,16:48]=2.5
    def observe(depth,z):
        return runtime.evaluate(depth,current_z_ned_m=z,frame_age_s=0.,gimbal_forward=True)[0]
    first=observe(wall,-1.)
    assert first.state.value=='HOLD'
    assert first.geometry['obstacle_extent_valid'] is False
    assert first.geometry['obstacle_far_north_m'] is None
    assert runtime._nominal_avoidance_altitude_m==1.
    observe(wall,-1.); observe(wall,-1.)
    assert runtime.policy.active
    assert observe(np.full((48,64),8.,dtype=np.float32),-2.).state.value=='CLEAR'
    assert not runtime.policy._roof_guard_active
    second=observe(wall,-4.)
    assert second.state.value=='HOLD' and not second.geometry['obstacle_extent_valid']
    assert runtime._nominal_avoidance_altitude_m==4.


@pytest.mark.parametrize('front_valid,blocked',[(True,False),(True,True),(False,False)])
def test_front_climb_actual_runtime_preserves_profile_reason_and_invalid_front(monkeypatch,front_valid,blocked):
    import jolgwa_uav.jetson_compute_service as service
    runtime=service.JetsonNativeDepthRuntime(VerticalObstacleAvoidanceCore(
        VerticalAvoidanceConfig(require_geometry=True,scenario_demo=True,front_climb_demo=True,
            trigger_samples=1,release_samples=1)))
    geometry=dict(geometry_valid=True,obstacle_extent_valid=False,roof_clearance_verified=False)
    monkeypatch.setattr(service,'_observe_native_geometry',lambda *a,**kw:(object(),geometry.copy()))
    depth=np.zeros((48,64),dtype=np.float32)
    if front_valid: depth[17:31,:]=2. if blocked else 8.
    decision,_=runtime.evaluate(depth,current_z_ned_m=-1.,frame_age_s=0.,gimbal_forward=True)
    assert decision.reason.startswith('scenario_front_detect_2m_pass_3m_v1:')
    assert decision.state.value==('STALE' if not front_valid else 'EVADE' if blocked else 'CLEAR')
    assert not decision.geometry['roof_clearance_verified']


def test_front_climb_relaxation_cannot_be_enabled_in_ordinary_profile():
    with pytest.raises(ValueError): VerticalAvoidanceConfig(front_climb_demo=True)
