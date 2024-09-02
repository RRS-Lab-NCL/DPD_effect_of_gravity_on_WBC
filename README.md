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
   
python3 main.py 0.05 0.00
