"""
PointCloud2 Binary Message Decoder.
Decodes standard ROS2 sensor_msgs/msg/PointCloud2 byte payloads into sanitized (N, 4) float32 arrays.
Supports variable field order, endianness, structured datatypes, and NaN/Inf filtering.
"""

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ROS2 PointField Datatypes (sensor_msgs/msg/PointField.msg)
INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
FLOAT32 = 7
FLOAT64 = 8

DTYPE_TO_NUMPY = {
    INT8: np.int8,
    UINT8: np.uint8,
    INT16: np.int16,
    UINT16: np.uint16,
    INT32: np.int32,
    UINT32: np.uint32,
    FLOAT32: np.float32,
    FLOAT64: np.float64,
}


@dataclass
class PointFieldDTO:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass
class PointCloud2DTO:
    """Mock/DTO representation of sensor_msgs/msg/PointCloud2 for environments without ROS2."""
    height: int
    width: int
    fields: List[PointFieldDTO]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool = False


def decode_pointcloud2_to_numpy(msg: Any) -> np.ndarray:
    """Decode a PointCloud2 message or DTO into a sanitized (N, 4) float32 NumPy array.

    Args:
        msg: sensor_msgs.msg.PointCloud2 instance or PointCloud2DTO instance.

    Returns:
        np.ndarray of shape (N, 4) with columns [x, y, z, intensity].
    """
    if msg is None:
        raise ValueError("PointCloud2 message cannot be None!")

    total_points = msg.height * msg.width
    if total_points == 0 or len(msg.data) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Extract field offsets
    field_map = {f.name: f for f in msg.fields}
    for req in ["x", "y", "z"]:
        if req not in field_map:
            raise ValueError(f"Required coordinate field '{req}' missing from PointCloud2 fields!")

    x_field = field_map["x"]
    y_field = field_map["y"]
    z_field = field_map["z"]
    int_field = field_map.get("intensity", field_map.get("i", field_map.get("reflectance", None)))

    # Use fast NumPy structured array if data matches point_step
    data_bytes = bytes(msg.data) if isinstance(msg.data, (bytes, bytearray, memoryview)) else bytes(msg.data)
    num_points = len(data_bytes) // msg.point_step

    if num_points == 0:
        return np.zeros((0, 4), dtype=np.float32)

    endian_prefix = ">" if getattr(msg, "is_bigendian", False) else "<"

    # Build structured dtype
    dtype_list = []
    current_offset = 0

    # Sort fields by offset
    sorted_fields = sorted(msg.fields, key=lambda f: f.offset)
    for f in sorted_fields:
        if f.offset > current_offset:
            dtype_list.append((f"padding_{current_offset}", f"V{f.offset - current_offset}"))
            current_offset = f.offset
        np_type = DTYPE_TO_NUMPY.get(f.datatype, np.float32)
        dtype_list.append((f.name, endian_prefix + np_type().dtype.str[1:]))
        current_offset = f.offset + np.dtype(np_type).itemsize

    if msg.point_step > current_offset:
        dtype_list.append((f"trailing_pad", f"V{msg.point_step - current_offset}"))

    try:
        raw_struct = np.frombuffer(data_bytes[:num_points * msg.point_step], dtype=np.dtype(dtype_list))
        x = raw_struct["x"].astype(np.float32)
        y = raw_struct["y"].astype(np.float32)
        z = raw_struct["z"].astype(np.float32)
        if int_field and int_field.name in raw_struct.dtype.names:
            intensity = raw_struct[int_field.name].astype(np.float32)
        else:
            intensity = np.zeros(num_points, dtype=np.float32)

        pts = np.column_stack([x, y, z, intensity])
    except Exception:
        # Fallback to manual byte slicing if structured unpacking fails
        pts_list = []
        for i in range(num_points):
            offset = i * msg.point_step
            x_val = struct.unpack_from(endian_prefix + "f", data_bytes, offset + x_field.offset)[0]
            y_val = struct.unpack_from(endian_prefix + "f", data_bytes, offset + y_field.offset)[0]
            z_val = struct.unpack_from(endian_prefix + "f", data_bytes, offset + z_field.offset)[0]
            int_val = struct.unpack_from(endian_prefix + "f", data_bytes, offset + int_field.offset)[0] if int_field else 0.0
            pts_list.append([x_val, y_val, z_val, int_val])
        pts = np.asarray(pts_list, dtype=np.float32)

    # Sanitize NaNs and Infs
    finite_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2])
    return pts[finite_mask]


def numpy_to_pointcloud2_dto(points: np.ndarray, frame_id: str = "velodyne") -> PointCloud2DTO:
    """Encode an (N, 4) float32 NumPy array into a PointCloud2DTO for testing and publishing."""
    if points.shape[0] == 0:
        return PointCloud2DTO(
            height=1, width=0,
            fields=[
                PointFieldDTO("x", 0, FLOAT32),
                PointFieldDTO("y", 4, FLOAT32),
                PointFieldDTO("z", 8, FLOAT32),
                PointFieldDTO("intensity", 12, FLOAT32),
            ],
            is_bigendian=False,
            point_step=16,
            row_step=0,
            data=b"",
            is_dense=True,
        )

    pts_float32 = points[:, :4].astype(np.float32)
    data_bytes = pts_float32.tobytes()

    return PointCloud2DTO(
        height=1,
        width=points.shape[0],
        fields=[
            PointFieldDTO("x", 0, FLOAT32),
            PointFieldDTO("y", 4, FLOAT32),
            PointFieldDTO("z", 8, FLOAT32),
            PointFieldDTO("intensity", 12, FLOAT32),
        ],
        is_bigendian=False,
        point_step=16,
        row_step=16 * points.shape[0],
        data=data_bytes,
        is_dense=True,
    )
