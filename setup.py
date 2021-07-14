from setuptools import setup
from setuptools import find_packages

long_description = open('README.md').read()

REQUIRED_PKGS = ['scipy',
                 'pandas',
                 'hampel',
                 'numpy',
                 'psutil',
                 'folium',
]

setup(
    name='NumMobility',
    packages=find_packages(exclude=('docs')),
    version='0.0.1',
    include_package_date=True,
    license='new BSD',
    python_requires='>=3.6',
    description='A Mobility-data Preprocessing Library using parallel computation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer='NumMobility Developers',
    maintainer_email='mobilitylab2021@gmail.com',
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 ],
    install_requires=REQUIRED_PKGS
    )
