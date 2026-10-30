"""Measurement metadata only: never influences routing, selection or optimization."""
import os,platform

def hardware(cuda=False):
 result=dict(host=platform.node(),cpu_threads=os.environ.get('SLURM_CPUS_PER_TASK'),job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),partition=os.environ.get('SLURM_JOB_PARTITION'))
 try:
  with open('/proc/cpuinfo') as f:
   for line in f:
    if line.startswith('model name'):result['cpu_model']=line.split(':',1)[1].strip();break
 except OSError:pass
 if cuda:
  import torch
  result.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,torch_cpu_threads=torch.get_num_threads())
  try:result.update(gpu_name=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
  except Exception:result['gpu_name']='unavailable'
 return result
