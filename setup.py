from setuptools import find_packages, setup

package_name = 'gvom'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Timothy Overbye',
    maintainer_email='Tim.Overbye@gmail.com',
    description='G-VOM: A GPU Accelerated Voxel Off-Road Mapping System',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)