# from https://github.com/gokayfem/ComfyUI_VLM_nodes/blob/main/install_init.py

import platform
import subprocess
import sys
import importlib.util
import re
import torch
import packaging.tags
from requests import get

def get_python_version():
    """Return the Python version in a concise format, e.g., '39' for Python 3.9."""
    version_match = re.match(r"3\.(\d+)", platform.python_version())
    if version_match:
        return "3" + version_match.group(1)
    else:
        return None

def get_system_info():
    """Gather system information related to NVIDIA GPU, CUDA version, AVX2 support, Python version, OS, and platform tag."""
    system_info = {
        'gpu': False,
        'cuda_version': None,
        'avx2': False,
        'python_version': get_python_version(),
        'os': platform.system(),
        'os_bit': platform.architecture()[0].replace("bit", ""),
        'platform_tag': None,
    }

    # Check for NVIDIA GPU and CUDA version
    if importlib.util.find_spec('torch'): 
        system_info['gpu'] = torch.cuda.is_available()
        if system_info['gpu']:
            system_info['cuda_version'] = "cu" + torch.version.cuda.replace(".", "").strip()
    
    # Check for AVX2 support
    if importlib.util.find_spec('cpuinfo'):
        try:
            # Attempt to import the cpuinfo module
            import cpuinfo
            
            # Safely attempt to retrieve CPU flags
            cpu_info = cpuinfo.get_cpu_info()
            if cpu_info and 'flags' in cpu_info:
                # Check if 'avx2' is among the CPU flags
                system_info['avx2'] = 'avx2' in cpu_info['flags']
            else:
                # Handle the case where CPU info is unavailable or does not contain 'flags'
                system_info['avx2'] = False
        except Exception as e:
            # Handle unexpected errors gracefully
            print(f"Error retrieving CPU information: {e}")
            system_info['avx2'] = False
    else:
        # Handle the case where the cpuinfo module is not installed
        print("cpuinfo module not available.")
        system_info['avx2'] = False
    # Determine the platform tag
    if importlib.util.find_spec('packaging.tags'):        
        system_info['platform_tag'] = next(packaging.tags.sys_tags()).platform

    return system_info

def latest_lamacpp():
    try:        
        response = get("https://api.github.com/repos/abetlen/llama-cpp-python/releases/latest")
        return response.json()["tag_name"].replace("v", "")
    except Exception:
        return "0.2.26"

def install_package(package_name, custom_command=None):
    if not package_is_installed(package_name):
        print(f"Installing {package_name}...")
        command = [sys.executable, "-m", "pip", "install", package_name, "--no-cache-dir"]
        if custom_command:
            command += custom_command.split()
        subprocess.check_call(command)
    else:
        print(f"{package_name} is already installed.")

def package_is_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

def install_llama(system_info):
    imported = package_is_installed("llama-cpp-python") or package_is_installed("llama_cpp")
    if imported:
        print("llama-cpp installed")
    else:
        lcpp_version = latest_lamacpp()
        base_url = "https://github.com/abetlen/llama-cpp-python/releases/download/v"
        avx = "AVX2" if system_info['avx2'] else "AVX"
        if system_info['gpu']:
            cuda_version = system_info['cuda_version']
            custom_command = f"--force-reinstall --no-deps --index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{avx}/{cuda_version}"
        else:
            custom_command = f"{base_url}{lcpp_version}/llama_cpp_python-{lcpp_version}-{system_info['platform_tag']}.whl"
        install_package("llama-cpp-python", custom_command=custom_command)

def main():
    system_info = get_system_info()
    install_llama(system_info)

if __name__ == "__main__":
    main()