import setuptools

with open('README.md') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='bertserini',
    version='0.0.2-2',
    packages=['bertserini',
              'bertserini.reader',
              'bertserini.retriever',
              'bertserini.experiments',
              'bertserini.utils',
              'bertserini.train'],
    url='https://github.com/rsvp-ai/bertserini',
    license='',
    author='bertserini',
    author_email='yuqing.xie@uwaterloo.ca',
    description='An end-to-end Open-Domain question answering system',
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
