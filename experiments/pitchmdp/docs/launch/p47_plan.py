"""Pure P47 launch planning metadata. Deliberately NO smoke, register, SSH or launch commands."""
from make_p47_config import build
from pitchmdp import direct_pitcher_id_registry as pid


def plan(family='EXP-P47-001'):
    if family != 'EXP-P47-001':
        raise ValueError('This preparation packet only plans EXP-P47-001')
    cfg = build()
    return {'family': family, 'design': cfg[pid.DESIGN_KEY], 'registered': False, 'execution_enabled': False,
            'arms': list(pid.P47_ARM_ORDER), 'seeds': list(pid.P47_SEEDS), 'fits': pid.P47_FITS,
            'gpu_uuid': pid.RETAINED_GPU, 'max_parallel': 1,
            'fit_projection_seconds': pid.P47_FAMILY_PROJECTION_SECONDS,
            'preflight_seconds_measured': None, 'launch_clock': None,
            'stop_grace_seconds': 900, 'aggregate_reserve_seconds': 1200,
            'requires_before_execution': cfg['execution']['blocked_until']}


if __name__ == '__main__':
    import json
    print(json.dumps(plan(), indent=2))
