import sys
import platform
import urllib.request
import json
import re

def get_wheel_url():
    try:
        import torch
    except ImportError:
        print("Error: torch not found. Install torch first.", file=sys.stderr)
        return None
    
    cuda_version = torch.version.cuda
    if not cuda_version:
        print("Error: PyTorch is CPU-only, but Flash Attention requires CUDA.", file=sys.stderr)
        return None
    
    # parse versions
    cu_parts = cuda_version.split(".")
    cu_tag = f"cu{cu_parts[0]}{cu_parts[1]}"
    
    torch_version = torch.__version__.split("+")[0]
    t_parts = torch_version.split(".")
    torch_tag = f"torch{t_parts[0]}.{t_parts[1]}"
    
    py_version = sys.version_info
    cp_tag = f"cp{py_version.major}{py_version.minor}"

    arch = platform.machine().lower()
    if arch == "x86_64" or arch == "amd64":
        arch_tag = "x86_64"
    elif arch == "aarch64" or arch == "arm64":
        arch_tag = "linux_arm64" if "linux" in sys.platform else arch
    else:
        arch_tag = arch

    import re
    
    url = "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases/tags/v0.0.0"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching github releases: {e}", file=sys.stderr)
        return None
        
    assets = [a['name'] for a in data.get('assets', [])]
    
    # We want to keep it to flash attention 2 for now
    # We will search for available cuda and torch versions matching python and arch
    # Then we fall back to the highest available that is <= our versions.
    
    best_wheel = None
    
    for minor_fallback in range(int(t_parts[1]), -1, -1):
        test_torch_tag = f"torch{t_parts[0]}.{minor_fallback}"
        
        # Also try exact cuda version first, then fallback to 124, 121, 118
        cu_fallbacks = [cu_tag]
        if cu_tag not in ["cu124", "cu121", "cu118"]:
            cu_fallbacks.extend(["cu124", "cu121", "cu118"])
            
        for test_cu_tag in cu_fallbacks:
            pattern = re.compile(rf"flash_attn-2\..*?\+{test_cu_tag}{test_torch_tag}-{cp_tag}-{cp_tag}.*{arch_tag}\.whl")
            matches = [a for a in assets if pattern.search(a)]
            
            if matches:
                matches.sort(reverse=True)
                best_wheel = matches[0]
                break
        if best_wheel:
            break
            
    if best_wheel:
        print(f"Found pre-built wheel: {best_wheel}", file=sys.stderr)
        return f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/{best_wheel}"
        
    print(f"No pre-built wheel found for {cu_tag}, {torch_tag}, {cp_tag}, {arch_tag}", file=sys.stderr)
    return None

if __name__ == "__main__":
    url = get_wheel_url()
    if url:
        print(url)
    else:
        sys.exit(1)
