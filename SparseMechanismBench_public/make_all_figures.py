"""Regenerate all project figures from CSV files."""
import subprocess, sys
for script in ['plot_results.py']:
    subprocess.run([sys.executable, script], check=True)
# Geometry/frozen-readout figures are created by their dedicated scripts.
print('Base figures regenerated. Run run_frozen_readout.py and run_representation_geometry.py for additional figures.')
