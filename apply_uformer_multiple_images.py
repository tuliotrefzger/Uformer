import os

dir = "SIDD_Small_sRGB_Only/Data"

for filename in os.listdir(dir):
    f = os.path.join(dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
