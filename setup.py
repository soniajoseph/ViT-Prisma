from setuptools import setup, find_packages


setup(
    name='vit-prisma',
    version='0.1.2',
    author='Sonia Joseph',
    author_email='soniamollyjoseph@gmail.com',
    description='A Vision Transformer library for mechanistic interpretability.',
    long_description=open('docs/README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/soniajoseph/vit-prisma',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={
    # Replace 'vit_prisma' with your package's actual name if different
    'vit_prisma': ['visualization/*.html', 'visualization/*.js'],
    # Add other patterns here as needed
},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    # install_requires=install_requires,
    extras_require={
        'dev': [
            'pytest>=6.0',
            # Other development dependencies...
        ],
    },
    keywords='vision-transformer, clip, multimodal, machine-learning, mechanistic interpretability',
    zip_safe=False,
)
