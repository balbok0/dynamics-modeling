from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from nav_msgs.msg import Odometry
import rospy
from scipy.spatial.transform import Rotation
import tqdm


def unroll_sequence_with_ts(twists: np.ndarray, ts: np.ndarray):
    start_state = np.zeros(twists.shape[1])

    seq = np.zeros((len(twists)+1, twists.shape[1]))
    seq[0] = start_state
    for t in range(len(twists)):
        curr_angle = seq[t,2]
        summand = twists[t,:]
        assert(summand.shape == (3,))
        world_summand = np.zeros_like(summand)
        world_summand[0] = np.cos(curr_angle) * summand[0] - np.sin(curr_angle) * summand[1]
        world_summand[1] = np.sin(curr_angle) * summand[0] + np.cos(curr_angle) * summand[1]
        world_summand[2] = summand[2]
        seq[t+1,:] = seq[t,:] + world_summand
    return seq


def plot_arr(axis: Axes, data: np.ndarray, t: np.ndarray, label: str = ""):
    # axis.plot(np.cumsum(data[:, 0]), np.cumsum(data[:, 1]))
    print(t.shape)
    print(data[:, 2].shape)
    t = t.squeeze()
    # theta = np.cumsum(data[:, 2])
    theta = np.cumsum(data[:, 2] * t)
    proj_x = np.array([np.ones_like(theta), np.tan(theta)]).T
    proj_x /= np.linalg.norm(proj_x, axis=1)[:, None]
    proj_y = proj_x[:, [1, 0]]
    proj_y[:, 0] *= -1



    plt = np.cumsum(t[:, None] * (data[:, 0][..., None] * proj_x + data[:, 1][..., None] * proj_y), axis=0)
    plt_x = plt[:, 0]
    plt_y = plt[:, 1]
    # seq = unroll_sequence_with_ts(data, t)

    # Plot less points to see distribution nicely
    plt_x = plt_x[::10]
    plt_y = plt_y[::10]

    axis.scatter(plt_x, plt_y, alpha=0.3)
    # axis.plot(seq[:, 0], seq[:, 1])
    # axis.plot(np.cumsum(t), data[:, 2])
    # print(np.cumsum(t)[-1])
    # print(t[:5])
    axis.set_title(label)


def parse_odom_msgs(msgs: List[Odometry], ts: List[rospy.Time]):

    assert len(msgs) == len(ts)

    time_diff = []
    poses = []

    vecs_along = []
    vecs_ortho = []

    for i in range(1, len(msgs)):

        prev_msg = msgs[i - 1]
        prev_ts = ts[i - 1]
        cur_msg = msgs[i]
        cur_ts = ts[i]

        # Time
        time_diff.append((cur_ts - prev_ts).to_sec())

        # Heading at last angle
        z_angle = Rotation(np.array([
            prev_msg.pose.pose.orientation.x,
            prev_msg.pose.pose.orientation.y,
            prev_msg.pose.pose.orientation.z,
            prev_msg.pose.pose.orientation.w,
        ])).as_euler('zyx')[0]
        heading_vec = np.array([1., np.tan(z_angle)])
        heading_vec /= np.linalg.norm(heading_vec)

        ortho_vec = heading_vec[[1, 0]]
        ortho_vec[0] *= -1

        # Get vector of displacement
        disp = np.array([
            cur_msg.pose.pose.position.x - prev_msg.pose.pose.position.x,
            cur_msg.pose.pose.position.y - prev_msg.pose.pose.position.y,
        ])

        # project
        proj_x = (heading_vec @ disp) * heading_vec
        proj_y = (ortho_vec @ disp) * ortho_vec

        # To get values for x and y dot projection with heading and ortho vectors
        proj_x = proj_x @ heading_vec
        proj_y = proj_y @ ortho_vec

        # Lastly get change in theta
        cur_z_angle = Rotation(np.array([
            cur_msg.pose.pose.orientation.x,
            cur_msg.pose.pose.orientation.y,
            cur_msg.pose.pose.orientation.z,
            cur_msg.pose.pose.orientation.w,
        ])).as_euler('zyx')[0]
        proj_theta = cur_z_angle - z_angle

        poses.append([proj_x, proj_y, proj_theta])

        vecs_along.append(heading_vec)
        vecs_ortho.append(ortho_vec)

    return np.array(poses), np.array(time_diff), np.array(vecs_along), np.array(vecs_ortho)


def reconstruct_from_odoms(odoms: np.ndarray, tsps: np.ndarray, start_pose: Optional[np.ndarray] = None):
    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])

    thetas = np.cumsum(odoms[:, 2]) + start_pose[2]

    along_vec = np.array([np.ones_like(thetas), np.tan(thetas)]).T
    along_vec /= np.linalg.norm(along_vec, axis=1, keepdims=True)

    ortho_vec = along_vec[:, [1, 0]]
    ortho_vec[:, 0] *= -1

    poses = np.cumsum(along_vec * odoms[:, 0, None] + ortho_vec * odoms[:, 1, None], axis=0)
    poses += start_pose[:2]

    return poses


def main():

    # raw_pred = np.load("data/y_pred.npy")
    # raw_true = np.load("data/y_true.npy")
    # ts = np.load("data/ts.npy", allow_pickle=True)
    # print(ts)

    # fig, ax = plt.subplots(1, 2)
    # plot_arr(ax[0], raw_pred[::2], ts[::2], "Pred")
    # plot_arr(ax[1], raw_true[::2], ts[::2], "True")
    # plt.show()

    # print("Here")
    import rosbag
    bag = rosbag.Bag("datasets/rzr_real/offtrail_7.5ms_OFPWS_6000_nv_ecc_reverse_2022-02-12-00-19-10.bag")
    plt_x = []
    plt_y = []
    plt_theta = []

    odom_msgs = []
    tsps = []
    for topic, msg, ts in tqdm.tqdm(bag.read_messages("/crl_rzr/odom")):
        odom_msgs.append(msg)
        tsps.append(ts)
        plt_x.append(msg.pose.pose.position.x)
        plt_y.append(msg.pose.pose.position.y)
        z_angle = Rotation(np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])).as_euler('zyx')[0]
        plt_theta.append(z_angle)

    odoms, ts, along_vecs, ortho_vecs = parse_odom_msgs(odom_msgs, tsps)
    start_pose = np.array(
        [
            plt_x[0],
            plt_y[0],
            plt_theta[0],
        ]
    )
    poses = reconstruct_from_odoms(odoms, ts, start_pose)

    fig, ax = plt.subplots(1, 2)
    # ax[0].scatter(poses[:, 0], poses[:, 1], alpha=0.1)
    # # ax[0].plot(np.cumsum(odoms[:, 2]) + start_pose[2])
    # ax[0].set_title("Transformed")
    # ax[1].scatter(plt_x, plt_y, alpha=0.1)
    # # ax[1].plot(plt_theta)
    # ax[1].set_title("Original")

    for t_x, t_y, pose, along, ortho in zip(plt_x[:-1], plt_y[:-1], poses, along_vecs, ortho_vecs):
        inner_prod = along @ ortho
        print(f"along scale: {along @ along}")
        print(f"ortho scale: {ortho @ ortho}")
        if inner_prod > 0:
            print(f"Inner product: {inner_prod}")

        ax[0].arrow(t_x, t_y, ortho[0], ortho[1], color="orange")
        ax[0].arrow(t_x, t_y, along[0], along[1], color="blue")

        ax[1].arrow(pose[0], pose[1], ortho[0], ortho[1], color="orange")
        ax[1].arrow(pose[0], pose[1], along[0], along[1], color="blue")

    # plt.scatter(plt_x, plt_y, alpha=0.1)
    plt.show()


if __name__ == "__main__":
    main()
