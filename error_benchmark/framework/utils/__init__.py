"""
Utility modules extracted from v4 injectors/validators.

- physics_utils: Force direction strategies (from ImpulseInjector)
- math_utils: Quaternion math (from PosePerturbInjector)
- rollout_utils: Post-injection trajectory tracking (from DropValidator)
"""

from .physics_utils import compute_force_direction, DIRECTION_STRATEGIES
from .math_utils import euler_to_quat, quat_multiply, quat_conjugate, quat_rotate_vector
from .rollout_utils import collect_rollout_stats
from .controlled_rng import ControlledRng
