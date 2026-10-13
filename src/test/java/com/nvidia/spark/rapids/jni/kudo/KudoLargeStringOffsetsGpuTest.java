/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Table;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KudoLargeStringOffsetsGpuTest {
  private static final int BYTES_PER_ROW = 1 << 20;

  private static HostColumnVector dense(boolean wide, int rows) {
    HostMemoryBuffer data = HostMemoryBuffer.allocate((long) rows * BYTES_PER_ROW, false);
    data.setMemory(0, data.getLength(), (byte) 'X');
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate((rows + 1L) * (wide ? 8 : 4), false);
    for (int row = 0; row <= rows; row++) {
      long at = (long) row * BYTES_PER_ROW;
      if (wide) offsets.setLong(row * 8L, at);
      else offsets.setInt(row * 4L, Math.toIntExact(at));
    }
    return new HostColumnVector(DType.STRING, rows, Optional.of(0L), data, null, offsets,
        Collections.emptyList(), wide ? DType.INT64 : DType.INT32);
  }
  private static void assertDense(Table table, int rows) {
    assertEquals(DType.INT64, table.getColumn(0).getOffsetsType());
    byte[] expected = new byte[BYTES_PER_ROW];
    Arrays.fill(expected, (byte) 'X');
    try (HostColumnVector host = table.getColumn(0).copyToHost()) {
      assertEquals(rows, host.getRowCount());
      assertEquals(DType.INT64, host.getOffsetsType());
      assertEquals((long) rows * BYTES_PER_ROW, host.getData().getLength());
      for (int row = 0; row < rows; row++) {
        assertEquals((long) row * BYTES_PER_ROW, host.getStartListOffset(row));
        assertEquals((row + 1L) * BYTES_PER_ROW, host.getEndListOffset(row));
        assertTrue(Arrays.equals(expected, host.getUTF8(row)), "Data mismatch in row " + row);
      }
    }
  }
  private static void write(KudoSerializer serializer, Path path, HostColumnVector input) throws Exception {
    try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(path), BYTES_PER_ROW)) {
      serializer.writeToStreamWithMetrics(new HostColumnVector[]{input}, out, 0,
          Math.toIntExact(input.getRowCount()));
    }
  }
  @Test void widePayloadOverTwoGiBStreamsAndRestoresOnGpu() throws Exception {
    Path path = Files.createTempFile("kudo-wide-payload-", ".bin");
    KudoSerializer serializer = new KudoSerializer(KudoStringOffsetsTest.schema(1));
    try {
      try (HostColumnVector input = dense(true, 2049)) {
        write(serializer, path, input);
        assertTrue(Files.size(path) > Integer.MAX_VALUE);
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path), BYTES_PER_ROW);
             KudoTable packed = KudoTable.from(in).get();
             KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{packed});
             Table table = merged.toTable()) {
          assertTrue(packed.getHeader().getTotalDataLength() > Integer.MAX_VALUE);
          assertThrows(ArithmeticException.class, packed.getHeader()::getTotalDataLen);
          assertDense(table, 2049);
        }
      }
    } finally {
      Files.deleteIfExists(path);
    }
  }
  @Test void mergingTwoNarrowTablesPromotesOffsetsWhenCharsExceedTwoGiB() throws Exception {
    Path path = Files.createTempFile("kudo-promote-offsets-", ".bin");
    KudoSerializer serializer = new KudoSerializer(KudoStringOffsetsTest.schema(1));
    try {
      try (HostColumnVector input = dense(false, 1025)) {
        write(serializer, path, input);
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path), BYTES_PER_ROW);
             KudoTable packed = KudoTable.from(in).get();
             KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{packed, packed});
             Table table = merged.toTable()) {
          assertEquals(DType.INT32, packed.getHeader().getOffsetsType(0));
          assertTrue(packed.getHeader().getTotalDataLength() < Integer.MAX_VALUE);
          assertEquals(DType.INT64, merged.getColumnInfoList()[0].getOffsetInfo().getOffsetsType());
          assertDense(table, 2050);
        }
      }
    } finally {
      Files.deleteIfExists(path);
    }
  }
}
