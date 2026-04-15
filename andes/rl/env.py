"""
Gymnasium environment wrapping ANDES time-domain simulation.
"""

import logging

import gymnasium
import numpy as np
from gymnasium import spaces

import andes

logger = logging.getLogger(__name__)


class AndesEnv(gymnasium.Env):
    """Gymnasium environment wrapping ANDES time-domain simulation.

    Wraps an ANDES power system case into a Gymnasium-compatible RL
    environment.  Each ``step()`` advances the simulation by ``dt``
    seconds and returns observations extracted from the DAE arrays.
    ``reset()`` uses ``TDS.reinit()`` (~1 ms) instead of reloading
    the case from disk.

    Parameters
    ----------
    case : str
        Case file path passed directly to ``andes.load()``.  Use
        ``andes.get_case()`` to resolve stock case keys.
    obs : list of tuple
        Observation spec.  Each element is ``(model, var)`` or
        ``(model, var, idx_list)``.  ``var.v_code`` determines the
        DAE array (``'x'``, ``'y'``, or ``'b'``).  When ``idx_list``
        is given, only those devices (external idx) are observed.
    acts : list of tuple
        Action spec.  Each element is ``(target, setpoint)`` or
        ``(target, setpoint, idx_list)``.  ``target`` may be a
        group name (e.g. ``'SynGen'``) or a model name (e.g.
        ``'TGOV1'``).  When ``idx_list`` is given, only those
        devices are controlled; otherwise all devices in the
        group or model are used.
    reward_fn : callable
        ``reward_fn(obs, action, env) -> float``.  Required.
    dt : float
        Simulation time per ``step()`` call in seconds.
    tf : float
        Episode end time in seconds.
    disturbance_fn : callable or None
        ``disturbance_fn(env) -> None``, called after each ``reinit``
        in ``reset()`` to inject episode-specific perturbations.
    crash_penalty : float
        Reward penalty applied when the simulation terminates due to
        instability (default ``-100.0``).  Set to ``0.0`` to disable.
    action_low : float or array-like
        Lower bound for the action space (default unbounded).
    action_high : float or array-like
        Upper bound for the action space (default unbounded).
    obs_low : float or array-like
        Lower bound for the observation space.
    obs_high : float or array-like
        Upper bound for the observation space.
    tds_config : dict or None
        Override TDS config keys (e.g. ``{'method': 'trap_adapt'}``).
        Applied after RL-optimized defaults (``no_tqdm=1``,
        ``save_every=0``).

    Examples
    --------
    >>> from andes.rl import AndesEnv
    >>> from andes.utils.paths import get_case
    >>> env = AndesEnv(
    ...     case=get_case('ieee14/ieee14_esst3a.xlsx'),
    ...     obs=[('GENROU', 'omega')],
    ...     acts=[('SynGen', 'paux')],
    ...     reward_fn=lambda obs, act, env: -float(sum((obs - 1.0)**2)),
    ...     dt=0.1, tf=20.0,
    ... )
    >>> obs, info = env.reset(seed=42)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        case,
        obs,
        acts,
        reward_fn,
        dt=0.1,
        tf=20.0,
        disturbance_fn=None,
        crash_penalty=-100.0,
        action_low=-np.inf,
        action_high=np.inf,
        obs_low=-np.inf,
        obs_high=np.inf,
        tds_config=None,
    ):
        super().__init__()

        if not callable(reward_fn):
            raise TypeError(
                "reward_fn must be a callable: "
                "reward_fn(obs, action, env) -> float"
            )

        # --- load, PFlow, TDS init ---
        ss = andes.load(case, default_config=True, no_output=True)
        if not ss.PFlow.run():
            raise RuntimeError(
                f"Power flow did not converge for case '{case}'."
            )

        # RL-optimized TDS defaults
        ss.TDS.config.no_tqdm = 1
        ss.TDS.config.save_every = 0

        # user overrides
        if tds_config is not None:
            for key, val in tds_config.items():
                if not hasattr(ss.TDS.config, key):
                    raise ValueError(
                        f"tds_config key '{key}' is not a valid "
                        f"TDS config attribute."
                    )
                setattr(ss.TDS.config, key, val)

        ss.TDS.init()
        self._ss = ss
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._dt = float(dt)
        self._tf = float(tf)
        self._reward_fn = reward_fn
        self._disturbance_fn = disturbance_fn
        self._crash_penalty = float(crash_penalty)
        self._step_count = 0

        # --- resolve observations ---
        self._obs_addrs = self._resolve_obs(obs)
        n_obs = sum(len(addrs) for _, addrs in self._obs_addrs)

        # --- resolve actions ---
        self._act_info = self._resolve_acts(acts)
        self._n_act = sum(len(info[3]) for info in self._act_info)

        # --- spaces ---
        self.observation_space = spaces.Box(
            low=np.float32(obs_low),
            high=np.float32(obs_high),
            shape=(n_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(action_low),
            high=np.float32(action_high),
            shape=(self._n_act,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ss(self):
        """The underlying ANDES System object."""
        return self._ss

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state.

        Uses ``TDS.reinit()`` for a fast (~1 ms) idempotent reset.
        """
        super().reset(seed=seed)

        self._ss.TDS.reinit()
        self._step_count = 0

        if self._disturbance_fn is not None:
            self._disturbance_fn(self)

        obs = self._get_obs()
        info = {'t': 0.0, 'step': 0}
        return obs, info

    def step(self, action):
        """Advance the simulation by ``dt`` seconds.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
            True if the simulation crashed.
        truncated : bool
            True if ``t >= tf``.
        info : dict
        """
        self._apply_action(action)

        # advance, clamping to tf
        t_next = min(float(self._ss.dae.t) + self._dt, self._tf)
        self._ss.TDS.config.tf = t_next
        success = self._ss.TDS.run(no_summary=True)

        obs = self._get_obs()

        truncated = (self._tf - float(self._ss.dae.t)) < 0.5 * self._dt
        terminated = (not success) and (not truncated)

        reward = float(self._reward_fn(obs, action, self))
        if terminated:
            reward += self._crash_penalty

        self._step_count += 1
        info = {'t': float(self._ss.dae.t), 'success': success,
                'step': self._step_count}

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_obs(self, obs_spec):
        """Resolve observation spec to (v_code, addrs) pairs.

        Returns
        -------
        list of (str, np.ndarray)
            Each entry is ``(v_code, addrs)`` where ``v_code`` is
            ``'x'``, ``'y'``, or ``'b'``.
        """
        addrs_list = []
        for spec in obs_spec:
            if len(spec) == 3:
                model_name, var_name, idx_list = spec
            elif len(spec) == 2:
                model_name, var_name = spec
                idx_list = None
            else:
                raise ValueError(
                    f"Obs spec must be (model, var) or (model, var, idx_list), "
                    f"got {spec!r}"
                )

            model = getattr(self._ss, model_name, None)
            if model is None:
                raise ValueError(f"Model '{model_name}' not found.")
            var = getattr(model, var_name, None)
            if var is None:
                raise ValueError(
                    f"Variable '{var_name}' not found on model '{model_name}'."
                )

            if idx_list is not None:
                if isinstance(idx_list, (str, int, float, np.integer)):
                    idx_list = [idx_list]
                uids = model.idx2uid(idx_list)
                addrs = np.array(var.a)[uids]
            else:
                addrs = np.array(var.a)

            addrs_list.append((var.v_code, addrs.copy()))

        return addrs_list

    def _resolve_acts(self, acts_spec):
        """Resolve action spec with full init-time validation.

        Each element of *acts_spec* is ``(target, setpoint)`` or
        ``(target, setpoint, idx_list)``.  When *idx_list* is given,
        only those devices are controlled.

        Returns list of ``(kind, target, setpoint, idx_list)`` tuples.
        """
        act_info = []
        for spec in acts_spec:
            if len(spec) == 3:
                target, setpoint, user_idx = spec
                if isinstance(user_idx, (str, int, float, np.integer)):
                    user_idx = [user_idx]
                else:
                    user_idx = list(user_idx)
            elif len(spec) == 2:
                target, setpoint = spec
                user_idx = None
            else:
                raise ValueError(
                    f"Action spec must be (target, setpoint) or "
                    f"(target, setpoint, idx_list), got {spec!r}"
                )

            in_groups = target in self._ss.groups
            in_models = target in self._ss.models
            if in_groups and in_models:
                raise ValueError(
                    f"'{target}' exists as both a group and a model. "
                    f"Use the specific model name or group name."
                )

            if in_groups:
                group = self._ss.groups[target]
                setter_name = f'set_{setpoint}'
                if not hasattr(group, setter_name):
                    raise ValueError(
                        f"Group '{target}' has no method '{setter_name}'."
                    )
                idx_list = user_idx if user_idx is not None else group.get_all_idxes()
                if not idx_list:
                    raise ValueError(f"Group '{target}' has no devices.")

                # validate all devices — fail early for missing controllers
                getter = getattr(group, f'get_{setpoint}', None)
                if getter is not None:
                    bad = []
                    for idx in idx_list:
                        try:
                            getter(self._ss, idx)
                        except KeyError:
                            bad.append(idx)
                    if bad:
                        raise ValueError(
                            f"Setpoint '{setpoint}' unavailable for devices "
                            f"{bad} in group '{target}'. "
                            f"Ensure all devices have the required controller."
                        )

                act_info.append(('group', target, setpoint, idx_list))

            elif in_models:
                model = self._ss.models[target]
                if not hasattr(model, setpoint):
                    raise ValueError(
                        f"Model '{target}' has no attribute '{setpoint}'."
                    )
                idx_list = user_idx if user_idx is not None else list(model.idx.v)
                if not idx_list:
                    raise ValueError(f"Model '{target}' has no devices.")
                act_info.append(('model', target, setpoint, idx_list))

            else:
                raise ValueError(
                    f"'{target}' not found in system groups or models."
                )

        return act_info

    def _get_obs(self):
        """Extract observation vector from DAE arrays."""
        parts = []
        for v_code, addrs in self._obs_addrs:
            arr = getattr(self._ss.dae, v_code)
            parts.append(arr[addrs].astype(np.float32))
        return np.concatenate(parts)

    def _apply_action(self, action):
        """Apply action vector to the system."""
        action = np.asarray(action, dtype=np.float64).ravel()
        if action.shape[0] != self._n_act:
            raise ValueError(
                f"Action length {action.shape[0]} != expected {self._n_act}"
            )
        offset = 0
        for kind, target, setpoint, idx_list in self._act_info:
            n = len(idx_list)
            values = action[offset:offset + n]
            if kind == 'group':
                group = self._ss.groups[target]
                setter = getattr(group, f'set_{setpoint}')
                for idx, val in zip(idx_list, values):
                    setter(self._ss, idx, float(val))
            else:
                model = self._ss.models[target]
                for idx, val in zip(idx_list, values):
                    model.set(setpoint, idx, attr='v', value=float(val))
            offset += n
