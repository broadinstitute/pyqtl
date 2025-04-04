import subprocess
import os


def check_dependency(name):
    """"""
    e = subprocess.call(f"which {name}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if e != 0:
        raise RuntimeError(f"External dependency '{name}' not installed")


def refresh_gcs_token():
    """"""
    t = subprocess.check_output('gcloud auth application-default print-access-token',
                                shell=True).decode().strip()
    os.putenv('GCS_OAUTH_TOKEN', t)
