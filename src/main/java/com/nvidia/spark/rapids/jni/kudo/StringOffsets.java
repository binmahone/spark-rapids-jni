/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostMemoryBuffer;

final class StringOffsets {
  private StringOffsets() {}
  static long read(HostColumnVectorCore column, long index) {
    return read(column.getOffsets(), index * column.getOffsetsType().getSizeInBytes(), column.getOffsetsType());
  }
  static long read(HostMemoryBuffer buffer, long byteOffset, DType type) {
    return DType.INT64.equals(type) ? buffer.getLong(byteOffset) : buffer.getInt(byteOffset);
  }
}
