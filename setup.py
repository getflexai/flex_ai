# python setup.py sdist bdist_wheel
# twine check dist/*
# twine upload dist/*

from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="flex_ai",
    version="0.39",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author="Ariel Cohen",
    author_email="ariel042cohen@gmail.com",
    description="Flex AI client library",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arielcohen4/flex_ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
     entry_points={
        'console_scripts': [
            'flex_ai=flex_ai.cli:main',
        ],
    },
)
