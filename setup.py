import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'qwen'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seohyeongmi',
    maintainer_email='hyeongmiseo9@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qwen_node = qwen.qwen_node:main',
            'text_emb = qwen.text_emb:main',
            'osnet_node = qwen.osnet_node:main',
        ],
    },
)
