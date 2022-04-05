from distutils.core import setup

setup(
    name='rosbag_to_torch',
    version='0.0.1',
    description='Conversion of rosbags into sequences that can be made into pytorch datasets',
    author='Jakub Filipek',
    author_email='balbok@cs.washington.edu',
    packages=['rosbag_to_torch'],
    package_dir={'rosbag_to_torch': 'src'},
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
