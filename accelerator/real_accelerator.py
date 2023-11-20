import os
# from vllm.logger import init_logger

# logger = init_logger(__name__)

SUPPORTED_ACCELERATOR_LIST = ['cuda', 'xpu']

accelerator = None


def is_current_accelerator_supported():
    return get_accelerator().device_name() in SUPPORTED_ACCELERATOR_LIST


def get_accelerator():
    global accelerator
    if accelerator is not None:
        return accelerator

    accelerator_name = None
    set_method = None
    # 1. Detect whether there is override of DeepSpeed accelerators from environment variable.
    if "VLLM_ACCELERATOR" in os.environ.keys():
        accelerator_name = os.environ["VLLM_ACCELERATOR"]
        if accelerator_name == "xpu":
            try:
                import intel_extension_for_pytorch
            except ImportError as e:
                raise ValueError(
                    f"XPU_Accelerator requires intel_extension_for_pytorch, which is not installed on this system.")
        elif is_current_accelerator_supported():
            raise ValueError(f'VLLM_ACCELERATOR must be one of {SUPPORTED_ACCELERATOR_LIST}. '
                             f'Value "{accelerator_name}" is not supported')
        set_method = "override"

    # 2. If no override, detect which accelerator to use automatically
    if accelerator_name is None:
        # We need a way to choose among different accelerator types.
        # Currently we detect which accelerator extension is installed
        # in the environment and use it if the installing answer is True.
        # An alternative might be detect whether CUDA device is installed on
        # the system but this comes with two pitfalls:
        # 1. the system may not have torch pre-installed, so
        #    get_accelerator().is_available() may not work.
        # 2. Some scenario like install on login node (without CUDA device)
        #    and run on compute node (with CUDA device) may cause mismatch
        #    between installation time and runtime.

        if accelerator_name is None:
            try:
                import intel_extension_for_pytorch  # noqa: F401,F811 # type: ignore

                accelerator_name = "xpu"
            except ImportError as e:
                pass
        if accelerator_name is None:
            accelerator_name = "cuda"

        set_method = "auto detect"

    # 3. Set accelerator accordingly
    if accelerator_name == "cuda":
        from .cuda_accelerator import CUDA_Accelerator

        accelerator = CUDA_Accelerator()
    elif accelerator_name == "xpu":
        from .xpu_accelerator import XPU_Accelerator

        accelerator = XPU_Accelerator()

    # logger.info(f"Setting accelerator to {accelerator._name} ({set_method})")
    return accelerator


def set_accelerator(accel_obj):
    global accelerator
    # logger.info(f"Setting accelerator to {accel_obj._name} (model specified)")
    accelerator = accel_obj
