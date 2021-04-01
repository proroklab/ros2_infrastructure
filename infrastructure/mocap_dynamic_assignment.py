import rclpy
import socket
import struct
import numpy as np
import datetime
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.publisher import Publisher
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, PoseStamped
from infrastructure.agent_util import get_uuids
from typing import Dict


def pose_to_p(pose):
    return np.array([pose.position.x, pose.position.y, 0.0])


def pose_to_r(pose) -> R:
    return R.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )


class MocapDynamicAssignment(Node):
    DIST_REJECT_POSE = 1.0  # reject poses if their distance is further away than this

    def __init__(self):
        super().__init__("mocap_dynamic_assignment")

        self.real_poses: Dict[str, PoseStamped] = {}
        self.poses_repubs: Dict[str, Publisher] = {}

        self.declare_parameter(
            "n_agents",
            value=1,
        )
        n_agents = self.get_parameter(f"n_agents")._value
        assert n_agents > 0
        self.get_logger().info(f"nagents {n_agents}")

        while True:
            all_agents = get_uuids()
            if len(all_agents) == n_agents:
                break

        for uuid in all_agents:
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

            initial_pose = PoseStamped()
            initial_pose.header.stamp = self.get_clock().now().to_msg()
            initial_pose.pose.position.x = initial_position[0]
            initial_pose.pose.position.y = initial_position[1]
            initial_pose.pose.position.z = initial_position[2]
            initial_orientation_quat = initial_orientation.as_quat()
            initial_pose.pose.orientation.x = initial_orientation_quat[0]
            initial_pose.pose.orientation.y = initial_orientation_quat[1]
            initial_pose.pose.orientation.z = initial_orientation_quat[2]
            initial_pose.pose.orientation.w = initial_orientation_quat[3]
            self.real_poses[uuid] = initial_pose

            rigid_body_name = self.get_parameter(f"{uuid}_rigid_body_label")
            self.create_subscription(
                PoseStamped,
                f"/motion_capture_server/rigid_bodies/{rigid_body_name}/pose",
                self.update_pose,
                qos_profile=qos_profile_sensor_data,
            )

            self.poses_repubs[uuid] = self.create_publisher(
                PoseStamped, f"/{uuid}/pose", qos_profile=qos_profile_sensor_data
            )

    def update_pose(self, pose):
        dists = []
        dists_uuids = []
        for uuid, real_pose in self.real_poses.items():
            dist = np.linalg.norm(pose_to_p(real_pose.pose) - pose_to_p(pose.pose))
            if dist > self.DIST_REJECT_POSE:
                self.get_logger().debug(f"Reject pose at dist {dist} to {uuid}")
                continue

            dist_rot = (
                pose_to_r(real_pose.pose) * pose_to_r(pose.pose).inv()
            ).magnitude()
            dists.append(dist + dist_rot / 10)
            dists_uuids.append(uuid)

        if len(dists) < 1:
            return

        index_min = np.argmin(dists)
        self.real_poses[dists_uuids[index_min]] = pose
        self.poses_repubs[dists_uuids[index_min]].publish(pose)


def main(args=None):
    rclpy.init(args=args)
    publisher = MocapDynamicAssignment()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
