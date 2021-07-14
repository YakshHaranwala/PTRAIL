from setuptools import setup
from setuptools import find_packages

#long_description = open('README.md').read()

REQUIRED_PKGS = ['numpy',
                 'hampel',
                 'pandas',
                 'scipy',
                 'psutil',
                 'folium',
]

setup(
    name='Nummobility',
    packages=find_packages(),
    version='0.0.1',
    license='new BSD',
    python_requires='>=3.6',
    description='A Mobility-data Preprocessing Library using parallel computation.',
    long_description="Hi! This is the home page of NumMobility library.",
    long_description_content_type="text/markdown",
    maintainer='NumMobility Developers',
    maintainer_email='mobilitylab2021@gmail.com',
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 ],
    install_requires=REQUIRED_PKGS,
    url='https://github.com/YakshHaranwala/NumMobility.git',
    include_package_data=True,
)
