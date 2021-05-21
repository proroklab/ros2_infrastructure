# Released by rdb under the Unlicense (unlicense.org)
# Based on information from:
# https://www.kernel.org/doc/Documentation/input/joystick-api.txt
# https://gist.github.com/rdb/8864666#file-js_linux-py

import rclpy
from rclpy.node import Node
from ctrl_msgs.msg import RoboMasterControl
import os
import struct
import array
import numpy as np
import fcntl
from fcntl import ioctl

# These constants were borrowed from linux/input.h
axis_names = {
    0x00: "x",
    0x01: "y",
    0x02: "z",
    0x03: "rx",
    0x04: "ry",
    0x05: "rz",
    0x06: "throttle",
    0x07: "rudder",
    0x08: "wheel",
    0x09: "gas",
    0x0A: "brake",
    0x10: "hat0x",
    0x11: "hat0y",
    0x12: "hat1x",
    0x13: "hat1y",
    0x14: "hat2x",
    0x15: "hat2y",
    0x16: "hat3x",
    0x17: "hat3y",
    0x18: "pressure",
    0x19: "distance",
    0x1A: "tilt_x",
    0x1B: "tilt_y",
    0x1C: "tool_width",
    0x20: "volume",
    0x28: "misc",
}

button_names = {
    0x120: "trigger",
    0x121: "thumb",
    0x122: "thumb2",
    0x123: "top",
    0x124: "top2",
    0x125: "pinkie",
    0x126: "base",
    0x127: "base2",
    0x128: "base3",
    0x129: "base4",
    0x12A: "base5",
    0x12B: "base6",
    0x12F: "dead",
    0x130: "a",
    0x131: "b",
    0x132: "c",
    0x133: "x",
    0x134: "y",
    0x135: "z",
    0x136: "tl",
    0x137: "tr",
    0x138: "tl2",
    0x139: "tr2",
    0x13A: "select",
    0x13B: "start",
    0x13C: "mode",
    0x13D: "thumbl",
    0x13E: "thumbr",
    0x220: "dpad_up",
    0x221: "dpad_down",
    0x222: "dpad_left",
    0x223: "dpad_right",
    # XBox 360 controller uses these codes.
    0x2C0: "dpad_left",
    0x2C1: "dpad_right",
    0x2C2: "dpad_up",
    0x2C3: "dpad_down",
}


class Joystick(Node):
    def __init__(self):
        super().__init__("mocap_static_remapping")
        # Iterate over the joystick devices.
        self.get_logger().info("Available devices:")

        for fn in os.listdir("/dev/input"):
            if fn.startswith("js"):
                self.get_logger().info(f"  /dev/input/{fn}")

        # We'll store the states here.
        self.axis_states = {}
        self.button_states = {}

        self.axis_map = []
        self.button_map = []

        # Open the joystick device.
        fn = "/dev/input/js0"
        self.get_logger().info(f"Opening {fn}...")
        self.jsdev = open(fn, "rb")
        fd = self.jsdev.fileno()
        flag = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)

        # Get the device name.
        # buf = bytearray(63)
        buf = array.array("B", [0] * 64)
        ioctl(self.jsdev, 0x80006A13 + (0x10000 * len(buf)), buf)  # JSIOCGNAME(len)
        js_name = buf.tobytes().rstrip(b"\x00").decode("utf-8")
        self.get_logger().info(f"Device name: {js_name}")

        # Get number of axes and buttons.
        buf = array.array("B", [0])
        ioctl(self.jsdev, 0x80016A11, buf)  # JSIOCGAXES
        num_axes = buf[0]

        buf = array.array("B", [0])
        ioctl(self.jsdev, 0x80016A12, buf)  # JSIOCGBUTTONS
        num_buttons = buf[0]

        # Get the axis map.
        buf = array.array("B", [0] * 0x40)
        ioctl(self.jsdev, 0x80406A32, buf)  # JSIOCGAXMAP

        for axis in buf[:num_axes]:
            axis_name = axis_names.get(axis, "unknown(0x%02x)" % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # Get the button map.
        buf = array.array("H", [0] * 200)
        ioctl(self.jsdev, 0x80406A34, buf)  # JSIOCGBTNMAP

        for btn in buf[:num_buttons]:
            btn_name = button_names.get(btn, "unknown(0x%03x)" % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0

        axes = ", ".join(self.axis_map)
        buttons = ", ".join(self.button_map)
        self.get_logger().info(f"{num_axes} axes found: {axes}")
        self.get_logger().info(f"{num_buttons} buttons found: {buttons}")

        self.pub_vel = self.create_publisher(RoboMasterControl, f"cmd_vel", 1)

        self.create_timer(1 / 100, self.run)

        self.ctrl = RoboMasterControl()
        self.multiplier = 1.0

    def run(self):
        evbuf = self.jsdev.read(8)
        if evbuf:
            time, value, type, number = struct.unpack("IhBB", evbuf)

            if type & 0x80:
                self.get_logger().debug("(initial)")

            if type & 0x01:
                button = self.button_map[number]
                if button:
                    self.button_states[button] = value
                    if value:
                        self.get_logger().debug("%s pressed" % (button))
                        if button == "tr":
                            self.multiplier += 0.5
                        if button == "tl":
                            self.multiplier -= 0.5
                        self.multiplier = np.clip(self.multiplier, 0.5, 3.0)
                        self.get_logger().info(f"multiplier is {self.multiplier}")
                    else:
                        self.get_logger().debug("%s released" % (button))

            if type & 0x02:
                axis = self.axis_map[number]
                if axis:
                    fvalue = value / 32767.0
                    self.axis_states[axis] = fvalue
                    self.get_logger().debug("%s: %.3f" % (axis, fvalue))
                    if axis == "x":
                        self.ctrl.vy = fvalue * self.multiplier
                    elif axis == "y":
                        self.ctrl.vx = -fvalue * self.multiplier
                    elif axis == "rx":
                        self.ctrl.omega = fvalue

        self.pub_vel.publish(self.ctrl)


def main(args=None):
    rclpy.init(args=args)
    publisher = Joystick()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
