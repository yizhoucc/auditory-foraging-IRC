from irc import __version__
from setuptools import setup, find_packages

setup(
    name="irc",
    version=__version__,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=['gym', 'stable-baselines3', 'jarvis'],
)
