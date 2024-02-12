import subprocess

celery_command = ['celery', 'worker', '-A', 'utils.psomap_tasks_manager.celery', '--loglevel=info']

subprocess.run(celery_command, stdout=subprocess.PIPE)

