from setuptools import setup, find_namespace_packages

setup(
    name='vsdkx-model-yolo-torch',
    url='https://github.com/natix-io/vsdkx-model-yolo-torch.git',
    author='Helmut',
    author_email='helmut@natix.io',
    namespace_packages=['vsdkx', 'vsdkx.model'],
    packages=find_namespace_packages(include=['vsdkx*']),
    dependency_links=[
        'git+https://gitlab+deploy-token-485942:VJtus51fGR59sMGhxHUF@gitlab.com/natix/cvison/vsdkx/vsdkx-core.git#egg=vsdkx-core'
    ],
    install_requires=[
        'vsdkx-core',
        'torch>=1.7.0',
        'opencv-python~=4.2.0.34',
        'torchvision>=0.8.1',
        'numpy==1.18.5',
        'pandas'
    ],
    version='1.0',
)
