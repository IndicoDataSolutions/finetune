import os
import subprocess
import warnings
from pathlib import Path

from setuptools.command.build_ext import build_ext

__all__ = ["OpsBuild"]


class OpsBuild(build_ext):
    """Run ``custom_ops/build.sh`` at build time."""

    BUILD_SCRIPT = Path(__file__).with_suffix("") / "build.sh"

    def run(self):
        script = str(self.BUILD_SCRIPT)
        # Ensure the script is executable; if not, try to chmod it.
        if not os.access(script, os.X_OK):
            try:
                os.chmod(script, 0o755)
            except OSError:
                warnings.warn(f"Could not mark build script as executable: {script}")

        result = subprocess.run(["sh", script])
        if result.returncode != 0:
            warnings.warn(
                "Failed to build the finetune memory‑management ops required for "
                "Scheduler. If you don't intend to use Scheduler you can safely "
                "ignore this message. To build the ops later run:\n    sh " + script,
                RuntimeWarning,
            )

        # Continue with the normal build_ext process even if the script fails
        super().run()
