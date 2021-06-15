from setuptools import setup, find_namespace_packages

setup(
    name='vsdkx-model-_yolo-torch',
    url='https://gitlab.com/natix/visiondeploy/aiconnector',
    author='Helmut',
    author_email='helmut@natix.io',
    # Needed to actually package something
    packages=find_namespace_packages(include=['vsdkx.model.*']),
    # Needed for dependencies
    dependency_links=[
        'git+https://gitlab+deploy-token-485942:VJtus51fGR59sMGhxHUF@gitlab.com/natix/cvison/vsdkx/vsdkx-core.git#egg=vsdkx-core'
    ],
    install_requires=[
        'vsdkx-core',
        'torch>=1.7.0',
        'opencv-python~=4.2.0.34',
        'torchvision>=0.8.1',
    ],
    # *strongly* suggested for sharing
    version='1.0',
)