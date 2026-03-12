import sounddevice as sd
import numpy as np

print("Testing all microphone devices...\n")

for i in [0, 1, 2, 6, 7, 8, 14, 15, 19, 24, 25]:
    try:
        d = sd.query_devices(i)
        if d['max_input_channels'] > 0:
            sr = int(d['default_samplerate'])
            sd.rec(1000, samplerate=sr, channels=1, device=i, dtype='float32')
            sd.wait()
            print(f"✅ Device {i} WORKS | SR:{sr} | {d['name']}")
    except Exception as e:
        print(f"❌ Device {i} FAILED | {str(e)[:50]}")
