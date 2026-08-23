from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'foveated_lidar_mapping'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy', 'torch', 'pyyaml'],
    zip_safe=True,
    maintainer='Atul',
    maintainer_email='atul@foveated-lidar.org',
    description='ROS2 Package for Foveated 2.5D LiDAR Mapping',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'foveated_mapping_node = foveated_lidar_mapping.foveated_mapping_node:main',
            'replay_node = foveated_lidar_mapping.replay_node:main',
        ],
    },
)
