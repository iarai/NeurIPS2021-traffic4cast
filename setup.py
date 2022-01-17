import setuptools

setuptools.setup(
    name="t4c21",
    version="0.0.1",
    author='Christian Eichenberger, Moritz Neun',
    description='',
    url="https://github.com/iarai/NeurIPS2021-traffic4cast",
    packages=[f't4c.{s}' for s in setuptools.find_packages()],
    package_dir={
        f't4c.{s}': s for s in setuptools.find_packages()
    },
    python_requires=">=3.8",
)
