import numpy as np
from typing import Optional


''' more advanced reconstrunt_from_odoms function. Not working yet. Some issue with dt I think.
def reconstruct_from_odoms(d_odom: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None, delay_steps: int = 1):
    """This function reconstructs the trajectory from odometry data.
    Odometry data is assumed to be in the form of [v_x, v_y, v_theta] or [v_x, v_theta],
    with dx and dy being in the robot frame.

    Args:
        d_odom (np.ndarray): A (n, 2) or (n, 3) array of odometry data.
        dt (np.ndarray): A (n,) array of differences in timestamps.
            It needs take delay_steps into account.
            This means that: dt[i] = t[i + delay_steps] - t[i]
        start_pose (Optional[np.ndarray], optional): A (3,) array of the starting pose (x, y, theta).
            Defaults to (0, 0, 0).
        delay_steps (int, optional): Number of steps taken for each prediction. Defaults to 1.

    Returns:
        _type_: _description_
    """
    assert len(d_odom.shape) == 2 and d_odom.shape[1] in {2, 3}, "d_odom must be a 2D array with 2 (dx, dtheta) or 3 (dx, dy, dtheta) columns"
    assert delay_steps >= 1, "Delay steps must be at least 1"

    # We expect dt to be a 1D array. It may be a (n, 1) array, in which case we'll reshape it to (n,).
    dt = dt.squeeze()

    d_x = d_odom[:, 0]
    if d_odom.shape[1] == 2:
        d_y = np.zeros_like(d_x)
        d_theta = d_odom[:, 1]
    else:
        # Shape == 3
        d_y = d_odom[:, 1]
        d_theta = d_odom[:, 2]

    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])

    # Rollout each continuous sequence seperately
    # Continuos sequence is meant by markovian chain where theta(i) = d_theta(i, j) + theta(j)
    size_rollout = len(d_x) // delay_steps
    delayed_size  = size_rollout * delay_steps
    delayed_shape = (size_rollout, delay_steps)

    # Reshape time series to (size_rollout, delay_steps)
    t_delayed = dt[:delayed_size].reshape(delayed_shape) // delay_steps

    thetas = np.reshape(d_theta[:delayed_size], delayed_shape)
    thetas = np.cumsum(thetas * t_delayed, axis=0) + start_pose[2]

    # Create vectors along and orthogonal to theta
    along_vec = np.concatenate((np.cos(thetas)[..., None], np.sin(thetas)[..., None]), axis=2)
    # Orthogonal vector is -sin(theta) along x and cos(theta) along y, so we can just use along
    ortho_vec = along_vec[..., [1, 0]]
    ortho_vec[..., 0] *= -1

    along_vals = np.reshape(d_x[:delayed_size], delayed_shape)
    ortho_vals = np.reshape(d_y[:delayed_size], delayed_shape)

    poses = np.cumsum(t_delayed[..., None] * (along_vec * along_vals[..., None] + ortho_vec * ortho_vals[..., None]), axis=0)

    poses = np.transpose(poses, (1, 0, 2)).reshape(-1, 2)
    poses += start_pose[:2]

    return np.hstack((poses, d_theta[:delayed_size, None]))
'''

def reconstruct_from_odoms(d_odom: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None, delay_steps: int = 1):
    """This function reconstructs the trajectory from odometry data.
    Odometry data is assumed to be in the form of [v_x, v_y, v_theta] or [v_x, v_theta],
    with dx and dy being in the robot frame.

    Args:
        d_odom (np.ndarray): A (n, 2) or (n, 3) array of odometry data.
        dt (np.ndarray): A (n,) array of differences in timestamps.
            It needs take delay_steps into account.
            This means that: dt[i] = t[i + delay_steps] - t[i]
        start_pose (Optional[np.ndarray], optional): A (3,) array of the starting pose (x, y, theta).
            Defaults to (0, 0, 0).
        delay_steps (int, optional): Number of steps taken for each prediction. Defaults to 1.

    Returns:
        np.ndarray: An (n, 3) array of poses. Columns correspond to (x, y, theta)
    """
    assert len(d_odom.shape) == 2 and d_odom.shape[1] in {2, 3}, f"d_odom must be a 2D array with 2 (dx, dtheta) or 3 (dx, dy, dtheta) columns. Instead it is of shape {d_odom.shape}"
    assert delay_steps >= 1, "Delay steps must be at least 1"

    # If d_odom has 2 columns add a column of zeros in the middle (for dy)
    if d_odom.shape[1] == 2:
        tmp = np.zeros((len(d_odom), 3))
        tmp[:, [0, 2]] = d_odom
        d_odom = tmp

    # We expect dt to be a 1D array. It may be a (n, 1) array, in which case we'll reshape it to (n,).
    dt = dt.squeeze()

    delayed_d_odom = d_odom
    # DEBUG: This is only an approximation. There has to be a better way to do this. This is good enough with well-behaved/distributed (timewise) data tho.
    delayed_dt = dt / delay_steps

    if len(delayed_dt) <= 3:
        return None

    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])

    thetas = np.cumsum(delayed_d_odom[:, 2] * delayed_dt, axis=0) + start_pose[2]

    # Create vectors along and orthogonal to theta
    along_vec = np.concatenate((np.cos(thetas)[..., None], np.sin(thetas)[..., None]), axis=1)
    # Orthogonal vector is -sin(theta) along x and cos(theta) along y, so we can just use along
    ortho_vec = along_vec[..., [1, 0]]
    ortho_vec[..., 0] *= -1

    poses = np.cumsum(delayed_dt[..., None] * (along_vec * delayed_d_odom[:, 0, None] + ortho_vec * delayed_d_odom[:, 1, None]), axis=0)
    poses += start_pose[:2]

    result_delayed = np.hstack((poses, thetas[:, None]))

    return result_delayed


def reconstruct_from_acc(acc: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None, start_vel: Optional[np.ndarray] = None, delay_steps: int = 1):
    if start_pose is None:
        start_pose = np.zeros(3)
    if start_vel is None:
        start_vel = np.zeros(3)

    # Convert from (x'', theta'') to (x'', y'', theta'')
    if acc.shape[1] == 2:
        tmp = np.zeros((len(acc), 3))
        tmp[:, [0, 2]] = acc
        acc = tmp

    # dt can sometimes be (n, 1), so just for sanity check we'll make sure it's (n,).
    dt = dt.squeeze()

    # Apply delay steps
    dt = dt[::delay_steps]
    acc = acc[::delay_steps]


    # 1. Rollout thetas
    disp_v_theta = np.cumsum(acc[:, 2] * dt)
    theta_cum = np.cumsum((start_vel[2] + disp_v_theta) * dt)
    thetas = (theta_cum + start_pose[2]).squeeze()

    # 2. Rollout velocities
    along_vec = np.concatenate((np.cos(thetas)[..., None], np.sin(thetas)[..., None]), axis=1)
    ortho_vec = along_vec[..., [1, 0]]

    disp_v = np.cumsum(dt[..., None] * (along_vec * acc[:, 0, None] + ortho_vec * acc[:, 1, None]), axis=0)
    disp_v = np.hstack((disp_v, disp_v_theta[:, None]))

    v = disp_v + start_vel

    return np.cumsum(dt[..., None] * v, axis=0) + start_pose
