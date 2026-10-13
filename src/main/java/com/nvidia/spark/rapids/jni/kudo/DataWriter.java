/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;

import java.io.IOException;

/**
 * Output data writer for kudo serializer.
 */
public interface DataWriter {

  /**
   * Write int in network byte order.
   */
  void writeInt(int i) throws IOException;

  /** Writes a long in network byte order. */
  default void writeLong(long value) throws IOException {
    writeInt((int) (value >>> 32));
    writeInt((int) value);
  }

  /** Reserves space when the writer supports a contiguous Java buffer. */
  default void reserve(long size) throws IOException {
    if (size <= Integer.MAX_VALUE) reserve(Math.toIntExact(size));
  }

  /**
   * Reserve space in the buffer for the given size.
   */
  default void reserve(int size) throws IOException {}

  /**
   * Copy data from src starting at srcOffset and going for len bytes.
   *
   * @param src       where to copy from.
   * @param srcOffset offset to start at.
   * @param len       amount to copy.
   */
  void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException;

  void flush() throws IOException;

  /**
   * Copy part of byte array to this writer.
   */
  void write(byte[] arr, int offset, int length) throws IOException;
}
