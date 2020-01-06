from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="deepgroebner",
    version="0.0.1",
    author="Dylan Peifer",
    author_email="djp282@cornell.edu",
    description="Applications of reinforcement learning to Groebner basis computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dylanpeifer/deepgroebner",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

