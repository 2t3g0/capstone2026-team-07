import pytest
from jolgwa_uav.front_climb_profile import configure, PROFILE
from jolgwa_uav.vertical_avoidance import VerticalAvoidanceConfig, VerticalObstacleAvoidanceCore

@pytest.mark.parametrize('speed',[0.,.5])
@pytest.mark.parametrize('latency',[0.,.499])
def test_dedicated_trigger_is_two_meters_not_old_dynamic_standoff(speed,latency):
    base=VerticalAvoidanceConfig(trigger_distance_m=3.,release_distance_m=5.5,minimum_standoff_m=2.5)
    new=configure(base)
    core=VerticalObstacleAvoidanceCore(new)
    assert core.effective_trigger_distance_m(forward_speed_mps=speed,pipeline_latency_s=latency)==2.
    assert new.release_distance_m==4.5
    assert base.trigger_distance_m==3. and base.minimum_standoff_m==2.5

def test_production_app_applies_profile_even_if_old_distance_env_remains(monkeypatch):
    from jolgwa_uav.jetson_compute_service import create_app
    monkeypatch.setenv('JOLGWA_SCENARIO_PROFILE',PROFILE)
    monkeypatch.setenv('JOLGWA_D435I_TRIGGER_DISTANCE_M','3.0')
    monkeypatch.setenv('JOLGWA_D435_MINIMUM_STANDOFF_M','2.5')
    monkeypatch.setenv('JOLGWA_PHASE1_ENABLED','false')
    config=create_app().state.native_depth_runtime.policy.config
    assert config.trigger_distance_m==2. and config.max_dynamic_trigger_distance_m==2.
    assert config.minimum_standoff_m==1. and config.release_distance_m==4.5

def test_observe_uses_same_two_meter_threshold_and_center_loss_is_unknown(monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    from jolgwa_ros.scenario_camera_status import depth_statistics, classify_depth_perception
    monkeypatch.setenv('JOLGWA_SCENARIO_PROFILE',PROFILE)
    def sample(mm):
        depth=np.full((480,640),mm,dtype='<u2')
        return depth_statistics(NS(encoding='16UC1',height=480,width=640,step=1280,data=depth.tobytes(),is_bigendian=False))
    for mm,expected in [(2001,'CLEAR'),(2000,'BLOCKED'),(1999,'BLOCKED')]:
        s=sample(mm);t=s['perception_thresholds']
        assert t['trigger_distance_m']==2.
        assert classify_depth_perception(near_distance_m=s['front_near_m'],obstacle_fraction=s['front_obstacle_fraction'],valid_fraction=s['center_valid_fraction'],upper_near_distance_m=None,upper_obstacle_fraction=None,upper_valid_fraction=0.,lower_valid_fraction=0.,**t)[0]==expected
    assert sample(0)['center_valid_fraction']==0.
