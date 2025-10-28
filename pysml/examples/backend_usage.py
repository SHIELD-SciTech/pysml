# pysml/examples/backend_usage.py
import pysml

print("Available:", pysml.engine.get_available_devices())
pysml.engine.set_device("cpu")
print("Now on:", pysml.engine.get_device())

# Pick fastest if available
for pref in ("cuda:0", "xpu:0"):
    if pref in pysml.engine.get_available_devices():
        pysml.engine.set_device(pref)
        print("Switched to:", pysml.engine.get_device())
        break
