from machine import Pin, I2S
import time
import ustruct
import sounddevice as sd

# I2S pin setup (adjust if needed)
SCK_PIN = 14
WS_PIN = 15
SD_PIN = 13

# Audio parameters
SAMPLE_RATE = 16000
RECORD_SECONDS = 6

# Configure I2S for INMP441
audio_in = I2S(
    0,
    sck=Pin(SCK_PIN),
    ws=Pin(WS_PIN),
    sd=Pin(SD_PIN),
    mode=I2S.RX,
    bits=16,
    format=I2S.MONO,
    rate=SAMPLE_RATE,
    ibuf=40000
)

# Open a file on the Pico
file = open("recorded.raw", "wb")

print("Recording...")
buf = bytearray(1024)
start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < RECORD_SECONDS * 1000:
    num_bytes = audio_in.readinto(buf)
    if num_bytes:
        file.write(buf)
file.close()
audio_in.deinit()

file = open("Recorded.wav")