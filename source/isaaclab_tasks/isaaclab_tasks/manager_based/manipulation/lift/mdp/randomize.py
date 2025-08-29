# isaaclab_tasks/.../lift/mdp/randomize.py
import re
import torch


def randomize_gripper_init(
    env, env_ids, asset_cfg, joint_names, range_rel=(0.1, 0.9), zero_velocity=True
):
    robot = env.scene[asset_cfg.name]

    # ---- resolve DOF indices from names/regex ----
    names = list(robot.data.joint_names)
    idxs = []
    for pat in joint_names:
        if pat in names:
            idxs.append(names.index(pat))
        else:
            import re
            rx = re.compile(pat)
            idxs.extend([i for i, n in enumerate(names) if rx.search(n)])
    if not idxs:
        raise ValueError(f"No joints matched patterns: {joint_names}")
    dof_ids = torch.tensor(sorted(set(idxs)), dtype=torch.long, device=env.device)  # (K,)

    # ---- get joint position limits (support (D,2) or (E,D,2)) ----
    limits = robot.data.joint_pos_limits  # torch.Tensor
    if limits.ndim == 2:
        # (D, 2)
        lo_abs = limits[dof_ids, 0]  # (K,)
        hi_abs = limits[dof_ids, 1]  # (K,)
    elif limits.ndim == 3:
        # (E, D, 2) -> pick any env (limits are identical across envs)
        lo_abs = limits[0, dof_ids, 0]  # (K,)
        hi_abs = limits[0, dof_ids, 1]  # (K,)
    else:
        raise RuntimeError(f"Unexpected joint_pos_limits shape: {limits.shape}")

    # shrink to a relative window inside each DOF’s range
    span = hi_abs - lo_abs
    lo = lo_abs + range_rel[0] * span        # (K,)
    hi = lo_abs + range_rel[1] * span        # (K,)

    # ---- sample per-env joint positions ----
    N = env_ids.numel()
    K = dof_ids.numel()
    r = torch.rand((N, K), device=env.device)            # (N, K)


    fixed_pos = (lo_abs + hi_abs) * 0.5  # midpoint
    new_pos = fixed_pos.unsqueeze(0).repeat(env_ids.numel(), 1)  # (N, K)
    #new_pos = lo.unsqueeze(0) + r * (hi - lo).unsqueeze(0)  # (N, K)


    # write into state tensors
    robot.data.joint_pos[env_ids[:, None], dof_ids] = new_pos
    if zero_velocity:
        robot.data.joint_vel[env_ids[:, None], dof_ids] = 0.0

    # sync back to sim
    robot.write_data_to_sim()


