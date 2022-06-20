import subprocess


dir = "/home/k/Desktop/python scripts/youtube/"
file = open(dir + "result.log")

for line in file:
    subprocess.call([dir+"Untitled-1.sh", line])
    print("done")