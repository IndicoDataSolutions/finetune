import re
import logging

from tensorflow.python.client import device_lib
from tensorflow.python.framework import config


LOGGER = logging.getLogger("finetune")

def fp16_capable():
    gpus = config.list_physical_devices('GPU')
    if not gpus:
        return False
    gpu_details_list = [config.get_device_details(g) for g in gpus]
    major_cc, _ = gpu_details_list[0]["compute_capability"]
    if major_cc >= 7:
        return True
    if any(gpu_info["compute_capability"][0] >= 7 for gpu_info in gpu_details_list):
        LOGGER.warning(
            (
                "A GPU with float 16 support is available but it is not GPU:0,"
                " to take advantage of fp16 inference set this to the only available GPU."
            )
        )
    return False

def gpu_info(session_config=None):
    is_fp16_capable = fp16_capable()
    num_devices = device_lib.list_local_devices(session_config=session_config)
    return {
        "n_gpus": num_devices,
        "fp16_inference": is_fp16_capable
    }
            
if __name__ == "__main__":
    print(gpu_info())
