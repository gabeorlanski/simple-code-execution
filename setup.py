from setuptools import find_packages
from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")


setup(
    name="code_execution",
    author="Gabriel Orlanski",
    author_email="gabeorlanski@gmail.com",
    version="0.1.0",
    python_requires=">=3.10",
    description="A simple code execution library for Python",
    long_description=readme,
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=requirements,
)
