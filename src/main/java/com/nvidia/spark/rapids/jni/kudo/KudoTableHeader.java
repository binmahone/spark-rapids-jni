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

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostMemoryBuffer;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.util.Optional;
import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.Preconditions.ensureNonNegative;
import static java.lang.Math.toIntExact;
import static java.util.Objects.requireNonNull;

/** Metadata for KUD0 tables and KUD1 tables with wide lengths or STRING offsets. */
public final class KudoTableHeader {
  private static final int NARROW_MAGIC = 0x4B554430;
  private static final int WIDE_MAGIC = 0x4B554431;
  private final int offset;
  private final int numRows;
  private final long validityBufferLen;
  private final long offsetBufferLen;
  private final long totalDataLen;
  private final int numColumns;
  private final byte[] hasValidityBuffer;
  private final byte[] wideOffsets;
  private final boolean extended;

  public static Optional<KudoTableHeader> readFrom(DataInputStream in) throws IOException {
    int magic;
    try { magic = in.readInt(); }
    catch (EOFException end) { return Optional.empty(); }
    if (magic != NARROW_MAGIC && magic != WIDE_MAGIC) {
      throw new IllegalStateException("Unsupported Kudo format magic: " + magic);
    }
    boolean extended = magic == WIDE_MAGIC;
    int offset = in.readInt();
    int rows = in.readInt();
    long validity = extended ? in.readLong() : in.readInt();
    long offsets = extended ? in.readLong() : in.readInt();
    long data = extended ? in.readLong() : in.readInt();
    int columns = ensureNonNegative(in.readInt(), "numColumns");
    byte[] validityBits = new byte[lengthOfHasValidityBuffer(columns)];
    byte[] widthBits = new byte[validityBits.length];
    in.readFully(validityBits);
    if (extended) in.readFully(widthBits);
    return Optional.of(new KudoTableHeader(offset, rows, validity, offsets, data, columns,
        validityBits, widthBits, extended));
  }

  KudoTableHeader(int offset, int rows, int validity, int offsets, int data,
                 int columns, byte[] validityBits) {
    this(offset, rows, validity, offsets, data, columns, validityBits,
        new byte[lengthOfHasValidityBuffer(columns)], false);
  }

  KudoTableHeader(int offset, int rows, long validity, long offsets, long data,
                 int columns, byte[] validityBits, byte[] widthBits, boolean forceExtended) {
    this.offset = ensureNonNegative(offset, "offset");
    this.numRows = ensureNonNegative(rows, "numRows");
    this.validityBufferLen = ensureNonNegative(validity, "validityBufferLen");
    this.offsetBufferLen = ensureNonNegative(offsets, "offsetBufferLen");
    this.totalDataLen = ensureNonNegative(data, "totalDataLen");
    this.numColumns = ensureNonNegative(columns, "numColumns");
    requireNonNull(validityBits, "hasValidityBuffer");
    requireNonNull(widthBits, "wideOffsets");
    ensure(validityBits.length == lengthOfHasValidityBuffer(columns), "Invalid validity bitmap length");
    ensure(widthBits.length == validityBits.length, "Invalid offsets-width bitmap length");
    ensure(Math.addExact(validity, offsets) <= data, "Buffer lengths exceed total data length");
    this.hasValidityBuffer = validityBits;
    this.wideOffsets = widthBits;
    this.extended = forceExtended || data > Integer.MAX_VALUE || hasAnyWideOffsets(widthBits);
  }

  static boolean hasAnyWideOffsets(byte[] bits) {
    for (byte value : bits) if (value != 0) return true;
    return false;
  }

  /** Returns an int length; throws ArithmeticException if the value cannot fit. */
  public int getTotalDataLen() { return toIntExact(totalDataLen); }
  /** Returns the total payload length in bytes. */
  public long getTotalDataLength() { return totalDataLen; }
  public int getValidityBufferLen() { return toIntExact(validityBufferLen); }
  public long getValidityBufferLength() { return validityBufferLen; }
  public int getOffsetBufferLen() { return toIntExact(offsetBufferLen); }
  public long getOffsetBufferLength() { return offsetBufferLen; }
  public boolean isExtendedFormat() { return extended; }
  public int getNumRows() { return numRows; }
  public int getOffset() { return offset; }
  public int getNumColumns() { return numColumns; }
  public boolean hasValidityBuffer(int index) {
    return (hasValidityBuffer[index / 8] & (1 << (index % 8))) != 0;
  }
  public DType getOffsetsType(int index) {
    if (index < 0 || index >= numColumns) throw new IndexOutOfBoundsException("Column " + index);
    return (wideOffsets[index / 8] & (1 << (index % 8))) != 0 ? DType.INT64 : DType.INT32;
  }
  byte[] getHasValidityBuffer() { return hasValidityBuffer; }
  public int getSerializedSize() { return getSerializedSize(hasValidityBuffer.length, extended); }
  public static int getSerializedSize(int bitmapLength) { return getSerializedSize(bitmapLength, false); }
  static int getSerializedSize(int bitmapLength, boolean extended) {
    return Math.addExact(extended ? 40 : 28, Math.multiplyExact(bitmapLength, extended ? 2 : 1));
  }
  public int startOffsetOf(BufferType type) { return toIntExact(startOffsetOfLong(type)); }
  public long startOffsetOfLong(BufferType type) {
    switch (type) {
      case VALIDITY: return 0;
      case OFFSET: return validityBufferLen;
      case DATA: return Math.addExact(validityBufferLen, offsetBufferLen);
      default: throw new IllegalArgumentException("Unsupported buffer type: " + type);
    }
  }

  public void writeTo(DataWriter out) throws IOException {
    out.writeInt(extended ? WIDE_MAGIC : NARROW_MAGIC);
    out.writeInt(offset);
    out.writeInt(numRows);
    if (extended) {
      out.writeLong(validityBufferLen);
      out.writeLong(offsetBufferLen);
      out.writeLong(totalDataLen);
    } else {
      out.writeInt(toIntExact(validityBufferLen));
      out.writeInt(toIntExact(offsetBufferLen));
      out.writeInt(toIntExact(totalDataLen));
    }
    out.writeInt(numColumns);
    out.write(hasValidityBuffer, 0, hasValidityBuffer.length);
    if (extended) out.write(wideOffsets, 0, wideOffsets.length);
  }

  private static void setInt(HostMemoryBuffer buffer, long at, int value) {
    byte[] bytes = new byte[] {(byte) (value >>> 24), (byte) (value >>> 16),
        (byte) (value >>> 8), (byte) value};
    buffer.setBytes(at, bytes, 0, bytes.length);
  }
  private static void setLong(HostMemoryBuffer buffer, long at, long value) {
    setInt(buffer, at, (int) (value >>> 32));
    setInt(buffer, at + 4, (int) value);
  }
  public long writeTo(HostMemoryBuffer buffer, long at) {
    setInt(buffer, at, extended ? WIDE_MAGIC : NARROW_MAGIC);
    setInt(buffer, at + 4, offset);
    setInt(buffer, at + 8, numRows);
    long next = at + 12;
    for (long length : new long[] {validityBufferLen, offsetBufferLen, totalDataLen}) {
      if (extended) { setLong(buffer, next, length); next += 8; }
      else { setInt(buffer, next, toIntExact(length)); next += 4; }
    }
    setInt(buffer, next, numColumns);
    next += 4;
    buffer.setBytes(next, hasValidityBuffer, 0, hasValidityBuffer.length);
    next += hasValidityBuffer.length;
    if (extended) buffer.setBytes(next, wideOffsets, 0, wideOffsets.length);
    return at + getSerializedSize();
  }
  static int lengthOfHasValidityBuffer(int columns) {
    ensureNonNegative(columns, "numColumns");
    return toIntExact((columns + 7L) / 8);
  }
  @Override public String toString() {
    return "KudoTableHeader{offset=" + offset + ", rows=" + numRows +
        ", dataLength=" + totalDataLen + ", columns=" + numColumns +
        ", extended=" + extended + "}";
  }
}
