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
   cd DPD_effect_of_gravity_on_WBC
   
2. **Set Up the Environment**

   ```bash
   conda create -n py39 python=3.9
   conda activate py39

3. **Install Dependencies**

   ```bash
   cd src
   pip install -r requirements.txt

4. **Compiling the Cython File**

   ```bash
   export CFLAGS="-I /home/test/anaconda3/lib/python3.9/site-packages/numpy/core/include $CFLAGS"
   python setup.py build_ext --inplace

5. **Check the Build**
   After the build process, ensure that the compiled file (e.g., your_module_name.cpython-<version>-<platform>.so) is present in the project directory.

6. **Run the simulation**

   ```bash
   python3 main.py body_force = <float_value> gravity = <float_value>
