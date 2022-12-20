import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='standardfunctions',
    version='1.0.1 l',
    author='Yannik Brune',
    author_email='yannik.brune@tu-dortmund.de',
    description='collection of functions to analyse data in csv, spe and img format',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Gamer317/standardfunctions',
    project_urls = {
        "Bug Tracker": "https://github.com/Gamer317/standardfunctions/issues"
    },
    license='GNU GPLv3',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires = ['scipy',
                        'numpy',
                        'uncertainties',
                        'opencv-python',
                        'pandas',
                        'spe2py',
                        'networkx',
                        'matplotlib',
                        'toolz'
],
)