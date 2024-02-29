from setuptools import find_packages, setup
import os

package_name = 'gvom'

launch_files = [(os.path.join('share', package_name, 'launch'), 
                 [os.path.join('launch', file) for file in os.listdir('launch') if file.endswith('.py')])]

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        *launch_files,
        (os.path.join('share', package_name, 'config'), 
         [os.path.join('config', 'gvom_params.yaml')]),
    ],
    package_data={
        package_name: ['config/gvom_params.yaml'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Timothy Overbye',
    maintainer_email='Tim.Overbye@gmail.com',
    description='G-VOM: A GPU Accelerated Voxel Off-Road Mapping System',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gvom = gvom.gvom_ros2:main'
        ],
    },
)