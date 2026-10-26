import subprocess,json
from pathlib import Path
def submit(*args):return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip()
prep=submit('surface_prepare.sbatch')
run=submit('--array=0-16%2','--dependency=afterok:'+prep+',afterany:23511217','surface.sbatch')
report=submit('--dependency=afterany:'+run,'report.sbatch')
final=submit('--dependency=afterany:'+run+':23511835','report.sbatch')
out=dict(prepare=prep,surface=run,report=report,final_report=final)
Path('surface_jobs.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
