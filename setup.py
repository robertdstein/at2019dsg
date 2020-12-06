import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="at2019dsg",
    version="0.1.0",
    author="Robert Stein",
    author_email="robert.stein@desy.de",
    description="Package with core code and data for at2019dsg",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="astroparticle physics science astronomy neutrino",
    url="https://github.com/robertdstein/at2019dsg",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.6',
    install_requires=[
        "emcee",
        "corner",
        "matplotlib",
        "astropy",
        "scipy",
        "numpy>=1.17.0",
        "pandas",
        "requests",
        "catsHTM",
        "flarestack",
        "pyregion",
        "pytest",
        "pytest-cov",
        "nbval",
        "pylatex",
        "ffmpeg-python",
        "gammapy",
        "tqdm",
        "sjoert @ https://github.com/sjoertvv/sjoert/archive/3308c1afe111693da0821b6d8b24a0439f0a648c.zip",
        "k3match @ https://github.com/pschella/k3match/archive/master.zip"
    ],
    package_data={'at2019dsg': [
        'at2019dsg/data/*']},
    include_package_data=True
)

