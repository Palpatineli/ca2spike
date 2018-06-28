from setuptools import setup

setup(
    name='ca2spike',
    version='0.1',
    requires=['numpy', 'uifunc'],
    packages=['ca2spike'],
    entry_points={
        'gui_scripts': [
            'ca2spike=ca2spike.main:convert'
        ]
    },
)
