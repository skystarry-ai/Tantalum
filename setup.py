from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.install import install
import os
import stat

class PostInstall(install):
    def run(self):
        install.run(self)
        gateway = os.path.join(
            self.install_lib,
            "service", "tantalum-gateway"
        )
        if os.path.exists(gateway):
            os.chmod(gateway, os.stat(gateway).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

setup(cmdclass={"install": PostInstall})
