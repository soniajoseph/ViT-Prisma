from setuptools import setup, find_packages

# Function to read the requirements file
def read_requirements(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

# Read the requirements file
install_requires = read_requirements('requirements.txt')

setup(
    name='vit-planetarium',
    version='0.1.0',
    author='Sonia Joseph',
    author_email='soniamollyjoseph@gmail.com',
    description='A Vision Transformer library for mechanistic interpretability.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/vit-planetarium',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pytest>=6.0',
            # Other development dependencies...
        ],
    },
    keywords='vision-transformer machine-learning',
    zip_safe=False,
)
