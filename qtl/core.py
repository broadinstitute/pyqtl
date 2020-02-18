import subprocess


def check_dependency(name):
    e = subprocess.call('which '+name, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if e!=0:
        raise RuntimeError('External dependency \''+name+'\' not installed')
