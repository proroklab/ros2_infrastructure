from setuptools import setup

package_name = 'infrastructure'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jan',
    maintainer_email='jb2270@cam.ac.uk',
    description='System-wide infrastructure (motion capture, stuff concerning all lab and all robots)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "mocap_static_remapping = infrastructure.mocap_static_remapping:main",
            "mocap_dynamic_assignment = infrastructure.mocap_dynamic_assignment:main",
            "mocap_velocity_estimator = infrastructure.mocap_velocity_estimator:main",
            "joystick = infrastructure.joystick:main",
        ],
    },
)
