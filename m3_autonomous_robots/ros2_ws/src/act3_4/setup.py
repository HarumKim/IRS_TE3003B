from setuptools import find_packages, setup

package_name = 'act3_4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mcl.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kim',
    maintainer_email='harumkim09@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'particle_generator = act3_4.particle_generator:main',
            'robot_navigator    = act3_4.robot_navigator:main',
        ],
    },
)
