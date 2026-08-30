import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parent
default_nb = repo / 'point_history_classification.ipynb'
nb = Path(sys.argv[1]) if len(sys.argv) > 1 else default_nb
if not nb.is_absolute():
	nb = repo / nb
python_exe = repo / 'venv' / 'Scripts' / 'python.exe'

cmd = [str(python_exe), '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', str(nb), '--inplace']
print('Running:', ' '.join(cmd))
result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
sys.exit(result.returncode)
