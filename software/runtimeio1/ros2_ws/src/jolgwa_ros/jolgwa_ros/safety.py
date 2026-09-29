from dataclasses import dataclass


@dataclass
class SafetyGate:
    """Independent gate checked before every PX4 input publication."""

    simulation_only: bool = True
    enable_px4_commands: bool = False
    allow_real_hardware: bool = False
    approved_mission_id: str = ""
    manual_override: bool = False
    connected: bool = False
    preflight_checks_pass: bool = False
    position_valid: bool = False
    armed: bool = False

    @property
    def command_output_enabled(self) -> bool:
        environment_allowed = self.simulation_only or self.allow_real_hardware
        return self.enable_px4_commands and environment_allowed

    def approve(self, mission_id: str, approved: bool) -> None:
        if approved:
            self.approved_mission_id = mission_id
        elif mission_id == self.approved_mission_id:
            self.approved_mission_id = ""

    def is_approved(self, mission_id: str) -> bool:
        return bool(mission_id) and mission_id == self.approved_mission_id

    def may_start_autonomy(self, mission_id: str) -> bool:
        return (
            self.command_output_enabled
            and self.is_approved(mission_id)
            and not self.manual_override
            and self.connected
            and self.preflight_checks_pass
            and self.position_valid
        )

    def may_stream_autonomy(self, mission_id: str) -> bool:
        return (
            self.command_output_enabled
            and self.is_approved(mission_id)
            and not self.manual_override
            and self.connected
            and self.position_valid
        )

    def may_stream_manual(self) -> bool:
        return (
            self.command_output_enabled
            and self.manual_override
            and self.armed
            and self.connected
            and self.position_valid
        )
