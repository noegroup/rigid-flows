#!/usr/bin/env python

import numpy as np

size = 2
print('repetitions:',size)

Tmin = 50
Tmax = 250
temp_steps = 15
temperatures = np.geomspace(Tmin, Tmax, temp_steps)
print(temperatures)

import subprocess
for temp in temperatures:
  subprocess.run(["sbatch", "../setup_ice_box.py", f"{size}", f"{temp}"])
