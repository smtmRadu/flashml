from setuptools import setup, find_packages

VERSION = "0.2"
setup(
    name="flashml", 
    version=VERSION,
    packages=find_packages(), 
    install_requires=[
        "matplotlib",
        "torch",
        "tqdm",
        "pillow",
    ],
    author='kbradu'
)