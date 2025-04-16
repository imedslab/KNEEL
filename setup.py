from setuptools import setup, find_packages

# Function to parse requirements files
def parse_requirements(filename):
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Skipping.")
        return []

# Read CPU and GPU specific requirements
requirements_cpu = parse_requirements('requirements_cpu.txt')
requirements_gpu = parse_requirements('requirements_gpu.txt')
# Read common requirements (if any)
# requirements_common = parse_requirements('requirements.txt')

setup(
    name='kneel',
    version='0.2',
    author='Aleksei Tiulpin',
    author_email='aleksei.tiulpin@oulu.fi',
    packages=find_packages(),
    # install_requires=requirements_common, # Use common requirements here if requirements.txt exists
    install_requires=(),
    extras_require={
        'gpu': requirements_gpu,
        'cpu': requirements_cpu,
        # You could also define an 'all' option if needed:
        # 'all': requirements_cpu + requirements_gpu
    },
    include_package_data=True,
    license='Creative Commons Attribution Non Commercial 4.0',
    long_description=open('README.md').read(), # Assumes README.md exists
)