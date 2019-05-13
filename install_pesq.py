import os
import subprocess
# subprocess.run(["cd", "python_pesq/pypesq"])
# subprocess.call('git submodule update', shell=True)
wd = os.getcwd()
os.chdir("pesq_lib/pypesq")
subprocess.call('python3 setup.py build_ext --inplace', shell=True)
os.chdir(wd)
subprocess.call('python3 test_pesq.py', shell=True)
