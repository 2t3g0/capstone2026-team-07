"""Production native runtime -> isolated DDS -> Action -> Controller -> Bridge.

Synthetic RGB-D/pose and confirmed event; no camera/GPU or serial output.
Uses the separate perception dependency environment, not ROS system Python.
"""
import pytest
from test_safety_contract import bridge
from test_scenario_chain import Chain, test_real_depth_dds_action_controller_bridge_memory_fc as run_chain
from jolgwa_uav.jetson_compute_service import JetsonNativeDepthRuntime


@pytest.mark.parametrize('height',[1.,2.])
@pytest.mark.parametrize('fault',['none','yaw_reset','battery_transient','battery_sustained','home_delayed','altitude_skew'])
def test_production_native_runtime_entire_scenario(height,fault,bridge,monkeypatch,tmp_path):
    original=Chain.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.native_runtime=JetsonNativeDepthRuntime(self.policy)
    monkeypatch.setattr(Chain,'__init__',init)
    run_chain(bridge,monkeypatch,tmp_path,height,fault)
