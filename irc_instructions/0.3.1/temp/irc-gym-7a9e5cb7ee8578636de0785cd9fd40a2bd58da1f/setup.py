from setuptools import setup, find_packages

with open('irc/VERSION.txt', 'r') as f:
    VERSION = f.readline().split('"')[1]

setup(
    name="irc",
    version=VERSION,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    package_data={'irc': ['VERSION.txt', 'rcParams.yaml']},
    install_requires=[
        'stable-baselines3', 'matplotlib', 'tqdm',
        'jarvis>=0.6.2',
    ],
)
