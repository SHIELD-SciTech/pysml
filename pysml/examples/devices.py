import sys, os
sys.path.append(os.getcwd())
import pysml

# Check available devices
available_devices = pysml.get_available_devices()
print(f"Available devices: {available_devices}")

# Check for specific device types
has_xpu = any('xpu' in device for device in available_devices)
has_cuda = any('cuda' in device for device in available_devices)
has_cpu = any('cpu' in device for device in available_devices)

print(f"CPU available: {has_cpu}")
print(f"XPU available: {has_xpu}")
print(f"CUDA available: {has_cuda}")

# Set device if available
if has_cuda:
    print("\n✓ CUDA is available!")
    pysml.set_device('cuda')
    print(f"Current device: {pysml.get_device()}")
elif has_xpu:
    print("\n✓ XPU is available!")
    pysml.set_device('xpu')
    print(f"Current device: {pysml.get_device()}")
else:
    print("\n→ Using CPU")
    pysml.set_device('cpu')
    print(f"Current device: {pysml.get_device()}")

# Get current backend
backend = pysml.get_backend()
print(f"Current backend: {backend}")