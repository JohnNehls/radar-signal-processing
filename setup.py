from setuptools import setup, find_packages

setup(
    name="radar-signal-processing",
    version="0.1.0",
    author="John Nehls",
    description="A module for creating range Doppler maps and general radar signal processing",
    long_description=open("README.org").read(),
    long_description_content_type="text/orgfile",
    url="https://github.com/JohnNehls/RadarSignalProcessing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)
