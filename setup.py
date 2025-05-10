from setuptools import setup, find_packages

setup(
    name="dollar_bars",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "numba>=0.50.0",
    ],
    author="Eric Ervin",
    author_email="eervin@blockforcecapital.com",  # Update this with your email
    description="A package for creating dollar bars from OHLCV data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eervin123/dollar-bars",  # Update this with your GitHub URL
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
