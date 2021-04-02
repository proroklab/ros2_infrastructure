import rclpy
import socket
import time
import numpy as np
import datetime
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.publisher import Publisher
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, PoseStamped
from infrastructure.agent_util import get_uuids
from typing import Dict, List


def pose_to_p(pose):
    return np.array([pose.position.x, pose.position.y, 0.0])


def pose_to_q(pose) -> np.ndarray:
    return np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )


def pose_to_r(pose) -> R:
    return R.from_quat(pose_to_q(pose))


class MocapDynamicAssignment(Node):
    DIST_REJECT_POSE = 1.0  # reject poses if their distance is further away than this

    def __init__(self):
        super().__init__("mocap_dynamic_assignment")

        self.declare_parameter(
            "n_agents",
            value=1,
        )
        n_agents = self.get_parameter(f"n_agents")._value
        assert n_agents > 0
        self.get_logger().info(f"nagents {n_agents}")

        while True:
            self.all_uuids = get_uuids()
            self.get_logger().info(
                f"Discovered {len(self.all_uuids)} agents (expecting {n_agents})"
            )
            if len(self.all_uuids) == n_agents:
                break
            self.get_logger().info("Retrying...")
            time.sleep(1)

        self.real_positions = np.zeros((n_agents, 3))
        self.real_orientations = np.zeros((n_agents, 4))

        self.poses_repubs: List[Publisher] = []
        for i, uuid in enumerate(self.all_uuids):
            self.get_logger().info(f"Remap {uuid}")

            self.declare_parameter(
                f"{uuid}_initial_position",
                value=[0.0, 0.0, 0.0],
            )
            self.declare_parameter(
                f"{uuid}_initial_orientation",
                value=[0.0, 0.0, 0.0],
            )
            self.declare_parameter(
                f"{uuid}_rigid_body_label",
                value=None,
            )

            initial_position = self.get_parameter(f"{uuid}_initial_position")._value
            initial_orientation = R.from_euler(
                "xyz", self.get_parameter(f"{uuid}_initial_orientation")._value
            )

            self.real_positions[i] = np.array(initial_position)
            self.real_orientations[i] = initial_orientation.as_quat()

            rigid_body_name = self.get_parameter(f"{uuid}_rigid_body_label")._value
            self.create_subscription(
                PoseStamped,
                f"/motion_capture_server/rigid_bodies/{rigid_body_name}/pose",
                self.update_pose,
                qos_profile=qos_profile_sensor_data,
            )

            self.poses_repubs.append(
                self.create_publisher(
                    PoseStamped, f"/{uuid}/pose", qos_profile=qos_profile_sensor_data
                )
            )

    def update_pose(self, pose):
        position = pose_to_p(pose.pose)
        dists = np.linalg.norm(self.real_positions - position, axis=1)

        # distance between two quats: https://math.stackexchange.com/a/90098
        orientation = pose_to_q(pose.pose)
        quat_norm = np.sum(self.real_orientations * orientation, axis=1)
        # clip the norm since it can be slightly outside this interval if the angles are identical
        # due to floating point arithmetic
        rot_dist = np.arccos(2 * np.clip(quat_norm, 0, 1) ** 2 - 1)

        cost = dists + rot_dist / 10
        index_min_cost = np.argmin(cost)
        if dists[index_min_cost] > self.DIST_REJECT_POSE:
            self.get_logger().debug(
                f"Reject pose at for dist {dists[index_min_cost]} to {self.all_uuids[index_min_cost]}"
            )
            return

        self.real_positions[index_min_cost] = position
        self.real_orientations[index_min_cost] = orientation[0]
        self.poses_repubs[index_min_cost].publish(pose)


def main(args=None):
    rclpy.init(args=args)
    publisher = MocapDynamicAssignment()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
