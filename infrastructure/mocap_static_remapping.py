import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from infrastructure.agent_util import get_uuids_fast
import functools


class MocapStaticRemap(Node):
    def __init__(self):
        super().__init__("mocap_static_remapping")
        self.poses_repubs = {}
        self.timer_refresh_pose_subs = self.create_timer(2.0, self.update_subscriptions)

    def update_subscriptions(self):
        for uuid in get_uuids_fast(self):
            if uuid in self.poses_repubs.keys():
                continue

            self.get_logger().info(f"Remap {uuid}")
            self.create_subscription(
                PoseStamped,
                f"/motion_capture_server/rigid_bodies/{uuid}/pose",
                functools.partial(self.update_pose, uuid),
                qos_profile=qos_profile_sensor_data,
            )
            self.poses_repubs[uuid] = self.create_publisher(
                PoseStamped, f"/{uuid}/pose", qos_profile=qos_profile_sensor_data
            )

    def update_pose(self, uuid: str, pose: PoseStamped):
        """Republish the raw mocap pose under a different name.
        If desired, a coordinate system transformation can be performed here.
        We can also consider to compute velocities here."""
        self.poses_repubs[uuid].publish(pose)


def main(args=None):
    rclpy.init(args=args)
    publisher = MocapStaticRemap()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
