# Effect of gravity on WBC
This project is designed to investigate the effects of gravity on WBC using Dissipative particle dynamics (DPD). It includes a Cython file that needs to be compiled before running the main code. This README provides instructions on how to set up the environment and compile the Cython file using `setup.py`.

## Requirements

Before you begin, ensure you have met the following requirements:

- Python 3.9 installed
- Cython installed (`pip install cython`)
- Pymp (Brings OpenMP-like functionality to Python; `pip install pymp-pypi`)
- A C compiler (like GCC on Linux/Mac or MSVC on Windows)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/RRS-Lab-NCL/DPD_effect_of_gravity_on_WBC.git
   cd repository
   
2. **Set Up the Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate

3. **Install Dependencies**

   ```bash
   cd /src
   pip install -r requirements.txt

3. **Compiling the Cython File**

   ```bash
   export CFLAGS="-I /home/test/anaconda3/lib/python3.xx/site-packages/numpy/core/include $CFLAGS"
   python setup.py build_ext --inplace

3. **Check the Build**
   After the build process, ensure that the compiled file (e.g., your_module_name.cpython-<version>-<platform>.so) is present in the project directory.

   
python3 main.py 0.05 0.00
