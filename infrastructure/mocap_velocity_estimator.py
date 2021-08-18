import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from infrastructure.agent_util import get_uuids_fast
import functools
import datetime
import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_p(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def pose_to_r(pose) -> R:
    return R.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )


def pose_to_t(pose):
    return datetime.timedelta(
        seconds=pose.header.stamp.sec, microseconds=pose.header.stamp.nanosec / 1000
    )


class MocapVelocityEstimator(Node):
    VELOCITY_RINGBUFFER_SIZE = 20

    def __init__(self):
        super().__init__("mocap_velocity_estimator")
        self.vel_pubs = {}
        self.last_poses = {}
        self.velocity_buffer = {}
        self.timer_refresh_pose_subs = self.create_timer(2.0, self.update_subscriptions)

    def update_subscriptions(self):
        for uuid in get_uuids_fast(self):
            if uuid in self.last_poses.keys():
                continue
            self.get_logger().info(f"Estimate {uuid}")
            self.create_subscription(
                PoseStamped,
                f"/motion_capture_server/rigid_bodies/{uuid}/pose",
                functools.partial(self.update_pose, uuid),
                qos_profile=qos_profile_sensor_data,
            )
            self.vel_pubs[uuid] = self.create_publisher(
                TwistStamped, f"/{uuid}/vel", qos_profile=qos_profile_sensor_data
            )
            self.last_poses[uuid] = None
            self.velocity_buffer[uuid] = []

    def update_pose(self, uuid: str, pose: PoseStamped):
        # if pose_time < self.last_pos_time:
        #    return

        if self.last_poses[uuid] is not None:
            # assert pose_time > self.last_pos_time
            dt = (pose_to_t(pose) - pose_to_t(self.last_poses[uuid])).total_seconds()
            v_lin = (pose_to_p(self.last_poses[uuid].pose) - pose_to_p(pose.pose)) / dt
            v_ang = (
                pose_to_r(self.last_poses[uuid].pose) * pose_to_r(pose.pose).inv()
            ).as_euler("xyz") / dt
            self.velocity_buffer[uuid].append(np.hstack([v_lin, v_ang]))
            if len(self.velocity_buffer[uuid]) > self.VELOCITY_RINGBUFFER_SIZE:
                self.velocity_buffer[uuid].pop(0)

            vel = np.mean(self.velocity_buffer[uuid], axis=0)
            vel_twist = TwistStamped(header=pose.header)
            vel_twist.twist.linear.x = vel[0]
            vel_twist.twist.linear.y = vel[1]
            vel_twist.twist.linear.z = vel[2]
            vel_twist.twist.angular.x = vel[3]
            vel_twist.twist.angular.y = vel[4]
            vel_twist.twist.angular.z = vel[5]
            self.vel_pubs[uuid].publish(vel_twist)

        self.last_poses[uuid] = pose


def main(args=None):
    rclpy.init(args=args)
    publisher = MocapVelocityEstimator()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
