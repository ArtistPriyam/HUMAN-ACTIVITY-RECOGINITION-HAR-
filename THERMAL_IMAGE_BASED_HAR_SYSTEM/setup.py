from setuptools import setup, find_packages

setup(
    name='har_heat_image',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'flask',
        'opencv-python',
        'numpy',
        'joblib',
        'ultralytics',
        # Add any others you use
    ],
    entry_points={
        'console_scripts': [],
    },
)
