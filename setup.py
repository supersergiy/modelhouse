import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modelhouse",
    version="0.0.0.1",
    author="Sergiy Popovych",
    author_email="sergiy.popovich@gmail.com",
    description="Easy model distribution for inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supersergiy/modelhouse",
    include_package_data=True,
    package_data={'': ['*.py']},
    install_requires=[
      'numpy'
    ],
    packages=setuptools.find_packages(),
)
