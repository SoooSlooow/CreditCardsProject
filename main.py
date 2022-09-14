import subprocess

subprocess.run(["dvc", "repro", "-f"], shell=True)
