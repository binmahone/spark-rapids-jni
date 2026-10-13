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

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import com.nvidia.spark.rapids.jni.schema.HostColumnsVisitor;

import java.util.ArrayDeque;
import java.util.Deque;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForValidityAlignment;
import static java.lang.Math.toIntExact;

/**
 * This class visits a list of columns and calculates the serialized table header.
 *
 * <p>
 * The columns are visited in post order, and for more details about the visiting process, please refer to
 * {@link HostColumnsVisitor}.
 * </p>
 */
class KudoTableHeaderCalc implements HostColumnsVisitor {
  private final SliceInfo root;
  private final int numFlattenedCols;
  private final byte[] bitset;
  private final byte[] wideOffsets;
  private long validityBufferLen;
  private long offsetBufferLen;
  private long dataOnlyLen;
  private int nextColIdx;

  private Deque<SliceInfo> sliceInfos = new ArrayDeque<>();

  KudoTableHeaderCalc(int rowOffset, int numRows, int numFlattenedCols) {
    this.root = new SliceInfo(rowOffset, numRows);
    this.dataOnlyLen = 0;
    sliceInfos.addLast(this.root);
    this.bitset = new byte[(numFlattenedCols + 7) / 8];
    this.wideOffsets = new byte[bitset.length];
    this.numFlattenedCols = numFlattenedCols;
    this.nextColIdx = 0;
  }

  public KudoTableHeader getHeader() {
    boolean extended = KudoTableHeader.hasAnyWideOffsets(wideOffsets);
    long paddedOffsetsSize = padForHostAlignment(offsetBufferLen);
    long paddedDataSize = padForHostAlignment(dataOnlyLen);
    int headerSize = KudoTableHeader.getSerializedSize(bitset.length, extended);
    long paddedValiditySize = padForValidityAlignment(validityBufferLen, headerSize);
    long total = Math.addExact(Math.addExact(paddedValiditySize, paddedOffsetsSize), paddedDataSize);
    if (!extended && total > Integer.MAX_VALUE) {
      extended = true;
      headerSize = KudoTableHeader.getSerializedSize(bitset.length, true);
      paddedValiditySize = padForValidityAlignment(validityBufferLen, headerSize);
      total = Math.addExact(Math.addExact(paddedValiditySize, paddedOffsetsSize), paddedDataSize);
    }
    return new KudoTableHeader(toIntExact(root.offset), toIntExact(root.rowCount),
        paddedValiditySize, paddedOffsetsSize, total, numFlattenedCols, bitset, wideOffsets, extended);
  }

  @Override
  public void preVisitStruct(HostColumnVectorCore col) {
    SliceInfo parent = sliceInfos.getLast();

    long validityBufferLength = 0;
    boolean includeValidity = col.hasValidityVector() && parent.rowCount > 0;
    if (includeValidity) {
      validityBufferLength = parent.getValidityBufferInfo().getBufferLength();
    }

    this.validityBufferLen += validityBufferLength;
    this.setHasValidity(includeValidity);
  }

  @Override
  public void visitStruct(HostColumnVectorCore col) {
    // NOOP
  }

  @Override
  public void preVisitList(HostColumnVectorCore col) {
    SliceInfo parent = sliceInfos.getLast();

    boolean includeValidity = col.hasValidityVector() && parent.rowCount > 0;

    long validityBufferLength = 0;
    if (includeValidity) {
      validityBufferLength = parent.getValidityBufferInfo().getBufferLength();
    }

    long offsetBufferLength = 0;
    if (col.getOffsets() != null && parent.rowCount > 0) {
      offsetBufferLength = (parent.rowCount + 1L) * Integer.BYTES;
    }

    this.validityBufferLen += validityBufferLength;
    this.offsetBufferLen += offsetBufferLength;

    this.setHasValidity(includeValidity);

    SliceInfo current;

    if (col.getOffsets() != null) {
      int start = col.getOffsets().getInt(parent.offset * Integer.BYTES);
      int end = col.getOffsets().getInt((parent.offset + parent.rowCount) * Integer.BYTES);
      int rowCount = end - start;
      current = new SliceInfo(start, rowCount);
    } else {
      current = new SliceInfo(0, 0);
    }

    sliceInfos.addLast(current);
  }

  @Override
  public void visitList(HostColumnVectorCore col) {
    sliceInfos.removeLast();
  }


  @Override
  public void visit(HostColumnVectorCore col) {
    if (DType.STRING.equals(col.getType()) && DType.INT64.equals(col.getOffsetsType())) {
      wideOffsets[nextColIdx / 8] |= (byte) (1 << (nextColIdx % 8));
    }
    SliceInfo parent = sliceInfos.peekLast();
    long validityBufferLen = dataLenOfValidityBuffer(col, parent);
    long offsetBufferLen = dataLenOfOffsetBuffer(col, parent);
    long dataBufferLen = dataLenOfDataBuffer(col, parent);

    this.validityBufferLen += validityBufferLen;
    this.offsetBufferLen += offsetBufferLen;
    this.dataOnlyLen += dataBufferLen;

    this.setHasValidity(col.hasValidityVector() && parent.rowCount > 0);
  }

  @Override
  public void done() {}

  private void setHasValidity(boolean hasValidityBuffer) {
    if (hasValidityBuffer) {
      int bytePos = nextColIdx / 8;
      int bitPos = nextColIdx % 8;
      bitset[bytePos] = (byte) (bitset[bytePos] | (1 << bitPos));
    }
    nextColIdx++;
  }

  private static long dataLenOfValidityBuffer(HostColumnVectorCore col, SliceInfo info) {
    if (col.hasValidityVector() && info.getRowCount() > 0) {
      return info.getValidityBufferInfo().getBufferLength();
    } else {
      return 0;
    }
  }

  private static long dataLenOfOffsetBuffer(HostColumnVectorCore col, SliceInfo info) {
    if (DType.STRING.equals(col.getType()) && info.getRowCount() > 0) {
      return (info.rowCount + 1L) * col.getOffsetsType().getSizeInBytes();
    } else {
      return 0;
    }
  }

  private static long dataLenOfDataBuffer(HostColumnVectorCore col, SliceInfo info) {
    if (DType.STRING.equals(col.getType())) {
      if (col.getOffsets() != null) {
        long startByteOffset = StringOffsets.read(col, info.offset);
        long endByteOffset = StringOffsets.read(col, (long) info.offset + info.rowCount);
        if (startByteOffset < 0 || endByteOffset < startByteOffset) {
          throw new IllegalArgumentException("Invalid string offsets");
        }
        return endByteOffset - startByteOffset;
      } else {
        return 0;
      }
    } else {
      if (col.getType().getSizeInBytes() > 0) {
        return col.getType().getSizeInBytes() * info.rowCount;
      } else {
        return 0;
      }
    }
  }
}
