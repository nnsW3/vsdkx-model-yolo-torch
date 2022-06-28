from setuptools import setup, find_namespace_packages

setup(
    name='vsdkx-model-yolo-torch',
    url='https://github.com/natix-io/vsdkx-model-yolo-torch.git',
    author='Helmut',
    author_email='helmut@natix.io',
    namespace_packages=['vsdkx', 'vsdkx.model'],
    packages=find_namespace_packages(include=['vsdkx*']),
    dependency_links=[
        'git+https://github.com/natix-io/vsdkx-core#egg=vsdkx-core'
    ],
    install_requires=[
        'vsdkx-core',
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'pandas',
        'tqdm>=4.41.0',
        'matplotlib>=3.2.2',
        'seaborn>=0.11.0',
        'numpy>=1.18.5'
    ],
    version='1.0',
)
