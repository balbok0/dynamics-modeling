from distutils.core import setup

from setuptools import find_packages

print(find_packages("src/package/rosbag2torch"))

setup(
    name='rosbag2torch',
    version='0.0.1',
    description='Conversion of rosbags into sequences that can be made into pytorch datasets',
    author='Jakub Filipek',
    author_email='balbok@cs.washington.edu',
    # packages=['rosbag2torch', 'rosbag2torch.*'],
    packages=find_packages("src/package"),
    package_dir={'rosbag2torch': 'src/package/rosbag2torch'},
    install_requires=[
        "numpy",
        "rospy-all",
        "rosbag",
        "pycryptodomex",
        "python-gnupg",
        "h5py",
        "tqdm",
        "scipy",
        "torch",
        "roslz4",
    ],
)
