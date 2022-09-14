from setuptools import setup

setup(
    name='kpps',
    version='1.0.0',
    description='Kris Plasma Particle Simulator (KPPS)',
    url='https://github.com/shuds13/pyexample',
    author='Kris Smedt',
    author_email='kristoffer@smedt.dk',
    license='BSD 2-clause',
    packages=['collocation',
              'initialisation',
              'model',
              'output'
              ],
    install_requires=['matplotlib',
                      'numpy',
                      'h5py'
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3+',
    ],
)