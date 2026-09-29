import json
import threading
import time
import uuid
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from jolgwa_interfaces.msg import MissionApproval, MissionProposal
from jolgwa_ros.scenario_contract import ScenarioRegistry
from jolgwa_ros.low_speed import AltitudeReferenceSynchronizer
from jolgwa_uav.mavlink_route_fc_link import RouteFcState
from test_safety_contract import bridge
from test_battery_landing import controller, install, brake_pair
from test_arm_home_transition import feed
from test_scenario import plan, command


@pytest.mark.parametrize('height',[1,2])
@pytest.mark.parametrize('metadata',['valid','absent','wrong_epoch','changed','retired'])
def test_real_bridge_write_checks_scenario_approval_at_final_gate(bridge,height,metadata):
    b,c=bridge,controller(height)
    b._scenario_enabled=True
    sync=b._altitude_reference_sync=AltitudeReferenceSynchronizer()
    r=b._route_state=RouteFcState(transport_epoch=sync.transport_epoch,home_stability_ns=0)
    cmd=c._active_command
    cmd.command=2; cmd.position_ned_m=[7.,0.,-float(height+1)]
    cmd.altitude_reference_epoch=sync.transport_epoch
    c._activate_output_contract(cmd)
    install(b,c)
    registry=b._scenario_registry=ScenarioRegistry()
    p=plan(float(height)); p['test_scenario']['transport_epoch']=sync.transport_epoch
    if metadata=='wrong_epoch': p['test_scenario']['transport_epoch']+=1
    if metadata!='absent': registry.proposal(NS(proposal_id='p',plan_json=json.dumps(p)))
    approval=NS(mission_id='m',proposal_id='p',approved=True)
    registry.approval(approval)
    if metadata=='changed':
        p['test_scenario']['home_z_ned_m']=1.
        registry.proposal(NS(proposal_id='p',plan_json=json.dumps(p)))
    if metadata=='retired': approval.approved=False; registry.approval(approval)
    base=time.monotonic_ns()
    home=dict(latitude=350000000,longitude=1290000000,altitude=40000,x=0.,y=0.,z=0.)
    assert r.accept('HOME_POSITION',home,1,1,base)
    for i in range(42):
        now=base+i*50_000_000
        with patch('time.monotonic_ns',return_value=now),patch('time.monotonic',return_value=now/1e9):
            feed(r,sync,now,1000+i*50)
            if i==1:
                assert r.lock_execution_home('m')[0]
                assert sync.set_home_reference(**r.home_correction_snapshot())
    with patch('time.monotonic_ns',return_value=now),patch('time.monotonic',return_value=now/1e9):
        brake_pair(b,now)
        b._stream_setpoint(r.snapshot(now),now)
    if metadata=='valid':
        assert b._transport.packets,b._last_output_detail
    else:
        assert not b._transport.packets
        assert 'scenario' in b._last_output_detail


def test_real_dds_approval_before_proposal_and_retirement():
    rclpy.init(args=[])
    suffix=uuid.uuid4().hex
    node=rclpy.create_node('scenario_isolated_dds_'+suffix)
    proposal_topic='/scenario_test/proposal_'+suffix
    approval_topic='/scenario_test/approval_'+suffix
    executor=MultiThreadedExecutor(num_threads=2); executor.add_node(node)
    registry=ScenarioRegistry(); event=threading.Event()
    node.create_subscription(MissionProposal,proposal_topic,
        lambda m:(registry.proposal(m),event.set()),10)
    node.create_subscription(MissionApproval,approval_topic,
        lambda m:(registry.approval(m),event.set()),10)
    pp=node.create_publisher(MissionProposal,proposal_topic,10)
    ap=node.create_publisher(MissionApproval,approval_topic,10)
    worker=threading.Thread(target=executor.spin,daemon=True); worker.start()
    try:
        for publisher in (pp,ap):
            deadline=time.monotonic()+3
            while publisher.get_subscription_count()==0 and time.monotonic()<deadline: time.sleep(.01)
            assert publisher.get_subscription_count()>0
        approval=MissionApproval(mission_id='m',proposal_id='p',approved=True)
        event.clear(); ap.publish(approval); assert event.wait(2.)
        assert registry.command_spec(command()) is None
        event.clear(); pp.publish(MissionProposal(proposal_id='p',plan_json=json.dumps(plan()))); assert event.wait(2.)
        assert registry.command_spec(command()) is not None
        approval.approved=False
        event.clear(); ap.publish(approval); assert event.wait(2.)
        assert registry.command_spec(command()) is None
    finally:
        executor.shutdown(timeout_sec=2); worker.join(2)
        node.destroy_node(); rclpy.shutdown()
