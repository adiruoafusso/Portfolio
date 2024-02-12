import subprocess

celery_command = ['flower', '-A', 'utils.psomap_tasks_manager.celery', '--port=5555']

subprocess.run(celery_command, stdout=subprocess.PIPE)
