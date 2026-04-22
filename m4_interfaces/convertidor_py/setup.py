from setuptools import find_packages, setup

package_name = 'convertidor_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kim',
    maintainer_email='harumkim09@gmail.com',
    description='Nodos ROS2 Python para el servicio de conversión de divisas',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'servidor_ros = convertidor_py.servidor_ros:main',
            'cliente_ros = convertidor_py.cliente_ros:main',
        ],
    },
)
