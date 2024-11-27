from setuptools import find_packages,setup

setup(
    name='mcqgenerator',
    version='0.1.0',
    author='waleed ahmed',
    author_email='ahmedwaleed0546@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'openai',"langchain","streamlit","python-dotenv","pyPDF2"],
    description='An AI mcq generator package',)