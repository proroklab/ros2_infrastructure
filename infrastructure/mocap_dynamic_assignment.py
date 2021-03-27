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
import functools
from typing import Dict


def pose_to_p(pose):
    return np.array([pose.position.x, pose.position.y, 0.0])


def pose_to_r(pose) -> R:
    return R.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )


class MocapDynamicAssignment(Node):
    POSE_ASSIGNMENT_FREQ = 100  # Hz

    def __init__(self):
        super().__init__("mocap_static_remap")

        self.real_poses: Dict[str, PoseStamped] = {}
        self.mocap_poses: Dict[str, PoseStamped] = {}
        self.poses_repubs: Dict[str, Publisher] = {}
        for uuid in get_uuids():
            self.get_logger().info(f"Remap {uuid}")

            self.real_poses[uuid] = PoseStamped()
            self.mocap_poses[uuid] = PoseStamped()

            self.create_subscription(
                PoseStamped,
                f"/motion_capture_server/rigid_bodies/{uuid}/pose",
                functools.partial(self.update_pose, uuid),
                qos_profile=qos_profile_sensor_data,
            )

            self.poses_repubs[uuid] = self.create_publisher(
                PoseStamped, f"/{uuid}/pose", qos_profile=qos_profile_sensor_data
            )

        self.create_timer(1 / self.POSE_ASSIGNMENT_FREQ, self.compute_assignment)

    def compute_assignment(self):
        """TODO:
        Compute correct assignment given the last known pose in real_poses
        and the poses most recently updated from the mocap system in mocap_poses
        """
        for uuid, pose in self.mocap_poses.items():
            self.real_poses[uuid] = pose
            self.poses_repubs[uuid].publish(pose)

    def update_pose(self, uuid, pose):
        """ Poses are updated asynchronously and assigned synchronously in the timer callback """
        self.mocap_poses[uuid] = pose


def main(args=None):
    rclpy.init(args=args)
    publisher = MocapDynamicAssignment()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
