import setuptools

setuptools.setup(
    name="t4c21",
    version="0.0.1",
    author="Christian Eichenberger, Moritz Neun",
    description="",
    url="https://github.com/iarai/NeurIPS2021-traffic4cast",
    packages=setuptools.find_packages(include=["t4c21", "t4c21.*"]),
    python_requires=">=3.8",
    install_requires=["overrides>=6.1.0", "h5py>=3.1.0", "pytorch-lightning>=1.4.2", "numpy>=1.19.5", "torch>=1.8.0", "psutil>=5.8.0"],
)
