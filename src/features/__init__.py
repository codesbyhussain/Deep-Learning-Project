"""Feature extraction (MultiROCKET), pooling, scaling, memmap storage."""

from src.features.memmap import create_memmap, open_memmap_read, write_array_to_memmap
from src.features.multirocket import (
    create_multirocket_transformer,
    extract_kernel_info,
    fit_multirocket,
    load_multirocket_transformer,
    save_multirocket_transformer,
    transform_multirocket_batched,
)
from src.features.pooling import build_dilation_pool_groups, structured_pool, structured_pooling_fallback
from src.features.scaling import fit_scaler, load_scaler, save_scaler, transform_with_scaler

__all__ = [
    "create_memmap",
    "write_array_to_memmap",
    "open_memmap_read",
    "create_multirocket_transformer",
    "fit_multirocket",
    "transform_multirocket_batched",
    "save_multirocket_transformer",
    "load_multirocket_transformer",
    "extract_kernel_info",
    "build_dilation_pool_groups",
    "fit_scaler",
    "transform_with_scaler",
    "save_scaler",
    "load_scaler",
    "structured_pool",
    "structured_pooling_fallback",
]
