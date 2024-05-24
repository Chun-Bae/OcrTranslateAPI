import subprocess
import asyncio

def docker_cp_sync(src: str, dest: str) -> subprocess.CompletedProcess:
    """
    Copies a file to/from a Docker container synchronously.
    :param src: Source path
    :param dest: Destination path
    :return: CompletedProcess instance
    """
    command = f"docker cp {src} {dest}"
    print(f"Running command: {command}")
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print(f"Failed to run command: {process.stderr.decode()}")
    else:
        print(f"Command output: {process.stdout.decode()}")
    return process

def docker_exec_sync(container: str, command: str) -> subprocess.CompletedProcess:
    """
    Executes a command inside a Docker container synchronously.
    :param container: Docker container name
    :param command: Command to execute
    :return: CompletedProcess instance
    """
    exec_command = f"docker exec {container} {command}"
    print(f"Running command: {exec_command}")
    process = subprocess.run(exec_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print(f"Failed to run command: {process.stderr.decode()}")
    else:
        print(f"Command output: {process.stdout.decode()}")
    return process

async def docker_cp(src: str, dest: str) -> subprocess.CompletedProcess:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, docker_cp_sync, src, dest)

async def docker_exec(container: str, command: str) -> subprocess.CompletedProcess:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, docker_exec_sync, container, command)

async def get_container_files(container_name: str, path: str) -> list:
    command = f"ls {path}"
    exec_result = await docker_exec(container_name, command)
    if exec_result.returncode != 0:
        print(f"Failed to list files in container: {exec_result.stderr.decode()}")
        return []
    files = exec_result.stdout.decode().split()
    return files