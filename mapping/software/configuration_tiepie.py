import libtiepie
import time
import sys

def config_TiePieScope(scp):
    try:
        scp.measure_mode = libtiepie.MM_BLOCK
        scp.sample_rate = 1e5  # 100 KHz
        scp.record_length = 50000  # 50000 samples
        scp.pre_sample_ratio = 0  # 0 %
        for ch in scp.channels:
            ch.enabled = False
            ch.range = 4  # 4 V
            ch.coupling = libtiepie.CK_DCV  # DC Volt
        ch2 = scp.channels[1]
        ch2.enabled = True
        ch2.range = 4  # 4 V
        ch2.coupling = libtiepie.CK_DCV  # DC Volt
        scp.trigger.timeout = 50e-3  # 100 ms
        for ch in scp.channels:
            ch.trigger.enabled = False
        ch = scp.channels[1]  # Ch 2
        ch.trigger.enabled = True
        ch.trigger.kind = libtiepie.TK_RISINGEDGE  # Rising edge
        ch.trigger.levels[0] = 0.5
        ch.trigger.hystereses[0] = 0.05  # 5 %
    except Exception as e:
        print(f'Exception: {e}')
        sys.exit(1)
    return scp