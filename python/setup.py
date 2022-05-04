from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='rz_linear',  # Required

    version='0.0.1',  # Required

    description='matrix multiply with compressed matrix using state-of-the-art ROBE-Z compression',  # Optional

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD-2 License'
    ],

    packages=find_packages(),  # Required

    python_requires='>=3.6, <4',

    install_requires=['torch', 'triton', 'pytest', 'numpy', 'pandas', 'tabulate'],  # Optional

    project_urls={  # Optional
        'Source': 'https://github.com/apd10/RzLinear'
    },

    include_package_data=True,

    long_description=long_description,

    long_description_content_type='text/markdown'
)
