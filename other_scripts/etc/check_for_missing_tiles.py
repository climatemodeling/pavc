import os
import glob

band = 'B12'
base = '/mnt/poseidon/remotesensing/arctic/data/rasters/S2SR/ak_arctic_summer'
d = f'{base}/{band}/2019-06-01_to_2019-08-31'

numbers = []
for file in sorted(glob.glob(f'{d}/GRIDCELL_*')):
    name = os.path.splitext(os.path.basename(file))[0]
    number = name.split('_')[1]
    numbers.append(int(number))

def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))

missing = missing_elements(numbers)
print(missing)