from setuptools import find_packages, setup

package_name = 'rov_practical'

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
    maintainer='eth',
    maintainer_email='mdether0056@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dddqn = rov_practical.DDDQN:main',
            'uw = rov_practical.unw:main',
            'tf = rov_practical.tftext:main',
        ],
    },
)