from setuptools import setup, find_packages

setup(
    name="ids_expt",
    version="0.1",
    author="Ramkrishna Acharya",
    description="A package for IDS experiments",
    long_description_content_type="text/markdown",
    url="https://github.com/q-viper/ids_expt",
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[],
)
