import re
import logging

from tensorflow.python.client import device_lib

LOGGER = logging.getLogger("finetune")

def gpu_info(session_config=None):
    def is_fp16_capable(device_desc):
        match = re.search(r"compute capability: (\d+)\.(\d+)", device_desc)
        if not match:
            return False
        return int(match.group(1)) >= 7
    num_gpus = 0
    fp16_capable = False
    for local_device in device_lib.list_local_devices(session_config=session_config):
        if local_device.device_type == "GPU":
            num_gpus += 1
            fp16 = is_fp16_capable(local_device.physical_device_desc)
            if fp16 and not fp16_capable:
                if num_gpus == 1:
                    fp16_capable = True
                else:
                    LOGGER.warning(
                        (
                            "A GPU with float 16 support is available but it is not GPU:0,"
                            " to take advantage of fp16 inference set this to the only available GPU. {}"
                        ).format(local_device.physical_device_desc)
                    )
    return {
        "n_gpus": num_gpus,
        "fp16_inference": fp16_capable
    }
            
if __name__ == "__main__":
    print(gpu_info())
