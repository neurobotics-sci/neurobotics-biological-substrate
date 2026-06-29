from setuptools import setup, find_packages

setup(
    name="brain-adam-eve",
    version="2026.3.28",
    description="Brain Adam & Eve: First Race of Silicon-Based Artificial Life Forms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kenneth Renshaw",
    author_email="kenneth@brain.ai",
    url="https://github.com/brain-brain/brain",
    license="AGPL-3.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21", "scipy>=1.7", "pyzmq>=22.0",
        "faiss-cpu>=1.7",
        "adafruit-circuitpython-ht16k33>=2.0",
        "RPi.GPIO>=0.7",
        "pyyaml>=6.0", "boto3>=1.26",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
    ],
    keywords="neuromorphic robotics artificial-life SBALF neurobotics humanoid",
)
