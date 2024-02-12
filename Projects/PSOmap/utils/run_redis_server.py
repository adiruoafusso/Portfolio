import os
import subprocess

main_app_working_directory = os.getcwd()

if os.path.exists(main_app_working_directory + '/redis-stable') is False:
    download_redis = ['curl', '-O', 'http://download.redis.io/redis-stable.tar.gz']
    untar_redis_archive = ['tar', 'xvzf', 'redis-stable.tar.gz']
    remove_redis_archive = ['rm', 'redis-stable.tar.gz']
    build_redis_server = ['make']
    for command in [download_redis, untar_redis_archive, remove_redis_archive, build_redis_server]:
        if command == build_redis_server:
            os.chdir(main_app_working_directory + '/redis-stable/')
        subprocess.run(command)
    os.chdir(main_app_working_directory)

redis_server_command = ['./redis-stable/src/redis-server']
subprocess.call(redis_server_command)
