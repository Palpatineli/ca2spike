from setuptools import setup

import subprocess

try:  # Try to create an rst long_description from README.md
    args = "pandoc", "--to", "rst", "README.md"
    long_description = subprocess.check_output(args)
    long_description = long_description.decode()
except Exception as error:
    print("README.md conversion to reStructuredText failed. Error:\n",
          error, "Setting long_description to None.")
long_description = None

setup(
    name="ca2spike",
    version="0.1.0",
    requires=["numpy", "uifunc"],
    packages=["ca2spike"],
    entry_points={
        "gui_scripts": [
            "ca2spike=ca2spike.main:convert"
        ]
    },
    description="convert calcium trace to spikes",
    long_description=long_description,
)
