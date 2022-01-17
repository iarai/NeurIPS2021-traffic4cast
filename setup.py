import setuptools

setuptools.setup(
    name="t4c21",
    version="0.0.1",
    author='Christian Eichenberger, Moritz Neun',
    description='',
    url="https://github.com/iarai/NeurIPS2021-traffic4cast",
    packages=setuptools.find_packages(include=['t4c21', 't4c21.*']),
    python_requires=">=3.8",
)