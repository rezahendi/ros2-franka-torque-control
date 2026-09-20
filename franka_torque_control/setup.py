from glob import glob

from setuptools import find_packages, setup

package_name = "franka_torque_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Seyed Reza Hendi",
    maintainer_email="rezahendi590@gmail.com",
    description="Torque control of a simulated Franka arm in MuJoCo through ROS 2.",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "sim_node = franka_torque_control.sim_node:main",
            "controller_node = franka_torque_control.controller_node:main",
            "trajectory_node = franka_torque_control.trajectory_node:main",
        ],
    },
)
