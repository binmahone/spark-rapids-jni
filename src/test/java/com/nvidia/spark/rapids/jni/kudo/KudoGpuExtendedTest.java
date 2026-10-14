/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KudoGpuExtendedTest {
  private static HostColumnVector wideStrings(String... values) {
    try (HostColumnVector narrow = HostColumnVector.fromStrings(values)) {
      HostMemoryBuffer offsets = HostMemoryBuffer.allocate((values.length + 1L) * 8, false);
      for (int i = 0; i < values.length; i++) {
        offsets.setLong(i * 8L, narrow.getStartListOffset(i));
      }
      offsets.setLong(values.length * 8L,
          values.length == 0 ? 0 : narrow.getEndListOffset(values.length - 1));
      HostMemoryBuffer data = narrow.getData();
      HostMemoryBuffer validity = narrow.getValidity();
      if (data != null) data.incRefCount();
      if (validity != null) validity.incRefCount();
      return new HostColumnVector(DType.STRING, values.length, Optional.of(narrow.getNullCount()),
          data, validity, offsets, Collections.emptyList(), DType.INT64);
    }
  }

  private static List<KudoTable> readGpuPartitions(DeviceMemoryBuffer[] serialized) throws Exception {
    List<KudoTable> tables = new ArrayList<>();
    try (HostMemoryBuffer data = HostMemoryBuffer.allocate(serialized[0].getLength(), false);
         HostMemoryBuffer offsets = HostMemoryBuffer.allocate(serialized[1].getLength(), false)) {
      data.copyFromDeviceBuffer(serialized[0]);
      offsets.copyFromDeviceBuffer(serialized[1]);
      int parts = Math.toIntExact(offsets.getLength() / 8 - 1);
      for (int part = 0; part < parts; part++) {
        long start = offsets.getLong(part * 8L);
        int length = Math.toIntExact(Math.min(offsets.getLong((part + 1L) * 8) - start, 4096));
        byte[] prefix = new byte[length];
        data.getBytes(prefix, 0, start, length);
        KudoTableHeader header = KudoTableHeader.readFrom(
            new DataInputStream(new ByteArrayInputStream(prefix))).get();
        assertTrue(header.isExtendedFormat());
        tables.add(new KudoTable(header,
            data.slice(start + header.getSerializedSize(), header.getTotalDataLength())));
      }
      return tables;
    } catch (Throwable failure) {
      for (KudoTable table : tables) table.close();
      throw failure;
    }
  }

  private static void assertStrings(Table table, DType width, String... expected) {
    assertEquals(width, table.getColumn(0).getOffsetsType());
    try (HostColumnVector host = table.getColumn(0).copyToHost()) {
      assertEquals(expected.length, host.getRowCount());
      for (int row = 0; row < expected.length; row++) {
        assertEquals(expected[row] == null, host.isNull(row), "Nullness in row " + row);
        if (expected[row] != null) assertEquals(expected[row], host.getJavaString(row), "Row " + row);
      }
    }
  }

  @Test void gpuWidePartitionsCanBeReadByCpuAndGpu() throws Exception {
    String[] values = {"中文", "", null, "tail"};
    try (HostColumnVector host = wideStrings(values);
         ColumnVector device = host.copyToDevice();
         Table input = new Table(device)) {
      DeviceMemoryBuffer[] packed = KudoGpuSerializer.splitAndSerializeToDevice(input, 0, 2, 2, 4);
      try (DeviceMemoryBuffer data = packed[0]; DeviceMemoryBuffer offsets = packed[1]) {
        Schema schema = KudoStringOffsetsTest.schema(1);
        List<KudoTable> tables = readGpuPartitions(packed);
        try {
          KudoSerializer serializer = new KudoSerializer(schema);
          try (KudoHostMergeResult merged = serializer.mergeOnHost(tables.toArray(new KudoTable[0]));
               Table restored = merged.toTable()) {
            assertStrings(restored, DType.INT64, values);
          }
        } finally {
          for (KudoTable table : tables) table.close();
        }
        try (Table restored = KudoGpuSerializer.assembleFromDeviceRaw(schema, data, offsets)) {
          assertStrings(restored, DType.INT64, values);
        }
      }
    }
  }

  @Test void gpuReadsMixedCpuNarrowAndWidePartitions() throws Exception {
    Schema schema = KudoStringOffsetsTest.schema(1);
    KudoSerializer serializer = new KudoSerializer(schema);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    long middle;
    try (HostColumnVector narrow = HostColumnVector.fromStrings("narrow", null);
         HostColumnVector wide = wideStrings("中文", "tail")) {
      serializer.writeToStreamWithMetrics(new HostColumnVector[]{narrow}, out, 1, 1);
      middle = out.size();
      serializer.writeToStreamWithMetrics(new HostColumnVector[]{wide}, out, 0, 2);
    }
    byte[] bytes = out.toByteArray();
    try (HostMemoryBuffer host = HostMemoryBuffer.allocate(bytes.length, false);
         HostMemoryBuffer positions = HostMemoryBuffer.allocate(24, false);
         DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(bytes.length);
         DeviceMemoryBuffer offsets = DeviceMemoryBuffer.allocate(24)) {
      host.setBytes(0, bytes, 0, bytes.length);
      positions.setLong(0, 0);
      positions.setLong(8, middle);
      positions.setLong(16, bytes.length);
      data.copyFromHostBuffer(host);
      offsets.copyFromHostBuffer(positions);
      try (Table restored = KudoGpuSerializer.assembleFromDeviceRaw(schema, data, offsets)) {
        assertStrings(restored, DType.INT64, null, "中文", "tail");
      }
    }
  }

  @Test void gpuRoundTripPreservesWideListChildAndNarrowListOffsets() throws Exception {
    HostColumnVector child = wideStrings("中文", null, "", "tail", "a", "b");
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate(16, false);
    offsets.setInt(0, 0);
    offsets.setInt(4, 2);
    offsets.setInt(8, 4);
    offsets.setInt(12, 6);
    try (HostColumnVector host = new HostColumnVector(DType.LIST, 3, Optional.of(0L),
             null, null, offsets, Collections.singletonList(child));
         ColumnVector device = host.copyToDevice();
         Table input = new Table(device)) {
      DeviceMemoryBuffer[] packed = KudoGpuSerializer.splitAndSerializeToDevice(input, 1, 2);
      Schema.Builder builder = Schema.builder();
      builder.addColumn(DType.LIST, "items").addColumn(DType.STRING, "element");
      try (DeviceMemoryBuffer data = packed[0]; DeviceMemoryBuffer positions = packed[1];
           Table restored = KudoGpuSerializer.assembleFromDeviceRaw(
               builder.build(), data, positions);
           HostColumnVector actual = restored.getColumn(0).copyToHost()) {
        assertEquals(DType.INT32, actual.getOffsetsType());
        assertEquals(0, actual.getStartListOffset(0));
        assertEquals(6, actual.getEndListOffset(2));
        assertEquals(DType.INT64, actual.getChildColumnView(0).getOffsetsType());
        String[] expected = {"中文", null, "", "tail", "a", "b"};
        for (int row = 0; row < expected.length; row++) {
          assertEquals(expected[row] == null, actual.getChildColumnView(0).isNull(row));
          if (expected[row] != null) {
            assertEquals(expected[row], actual.getChildColumnView(0).getJavaString(row));
          }
        }
      }
    }
  }

  @Test void gpuRoundTripPreservesNullStructParentAndWideStringChild() throws Exception {
    HostColumnVector child = wideStrings("中文", null, "", "tail");
    HostMemoryBuffer validity = HostMemoryBuffer.allocate(64, false);
    validity.setMemory(0, validity.getLength(), (byte) 0);
    validity.setByte(0, (byte) 0x0d);
    try (HostColumnVector host = new HostColumnVector(DType.STRUCT, 4, Optional.of(1L),
             null, validity, null, Collections.singletonList(child));
         ColumnVector device = host.copyToDevice();
         Table input = new Table(device)) {
      DeviceMemoryBuffer[] packed = KudoGpuSerializer.splitAndSerializeToDevice(input, 1, 3);
      Schema.Builder builder = Schema.builder();
      builder.addColumn(DType.STRUCT, "record").addColumn(DType.STRING, "value");
      try (DeviceMemoryBuffer data = packed[0]; DeviceMemoryBuffer positions = packed[1];
           Table restored = KudoGpuSerializer.assembleFromDeviceRaw(
               builder.build(), data, positions);
           HostColumnVector actual = restored.getColumn(0).copyToHost()) {
        assertEquals(4, actual.getRowCount());
        assertEquals(1, actual.getNullCount());
        assertFalse(actual.isNull(0));
        assertTrue(actual.isNull(1));
        assertFalse(actual.isNull(2));
        assertFalse(actual.isNull(3));
        assertEquals(DType.INT64, actual.getChildColumnView(0).getOffsetsType());
        assertEquals("中文", actual.getChildColumnView(0).getJavaString(0));
        assertTrue(actual.getChildColumnView(0).isNull(1));
        assertEquals("", actual.getChildColumnView(0).getJavaString(2));
        assertEquals("tail", actual.getChildColumnView(0).getJavaString(3));
      }
    }
  }

  @Test void gpuRoundTripPreservesNarrowColumnsInPayloadAboveTwoGiB() throws Exception {
    int rows = 1025;
    int bytesPerRow = 1 << 20;
    HostMemoryBuffer data = HostMemoryBuffer.allocate(rows * (long) bytesPerRow, false);
    data.setMemory(0, data.getLength(), (byte) 'x');
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate((rows + 1L) * 4, false);
    for (int row = 0; row <= rows; row++) offsets.setInt(row * 4L, row * bytesPerRow);
    try (HostColumnVector host = new HostColumnVector(DType.STRING, rows, Optional.of(0L),
             data, null, offsets, Collections.emptyList());
         ColumnVector device = host.copyToDevice();
         Table input = new Table(device, device)) {
      DeviceMemoryBuffer[] packed = KudoGpuSerializer.splitAndSerializeToDevice(input);
      try (DeviceMemoryBuffer payload = packed[0]; DeviceMemoryBuffer positions = packed[1];
           Table restored = KudoGpuSerializer.assembleFromDeviceRaw(
               KudoStringOffsetsTest.schema(2), payload, positions)) {
        assertTrue(payload.getLength() > Integer.MAX_VALUE);
        byte[] expected = new byte[bytesPerRow];
        Arrays.fill(expected, (byte) 'x');
        for (int column = 0; column < 2; column++) {
          assertEquals(DType.INT32, restored.getColumn(column).getOffsetsType());
          try (HostColumnVector output = restored.getColumn(column).copyToHost()) {
            assertEquals(rows * (long) bytesPerRow, output.getEndListOffset(rows - 1));
            for (int row = 0; row < rows; row++) {
              assertArrayEquals(expected, output.getUTF8(row), "Row " + row);
            }
          }
        }
      }
    }
  }

  @Test void truncatedExtendedPayloadIsRejectedBeforeAllocation() throws Exception {
    KudoTableHeader header = new KudoTableHeader(0, 1, 0L, 16L, 1024L,
        1, new byte[]{0}, new byte[]{1}, true);
    try (HostMemoryBuffer host = HostMemoryBuffer.allocate(header.getSerializedSize(), false);
         HostMemoryBuffer positions = HostMemoryBuffer.allocate(16, false);
         DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(host.getLength());
         DeviceMemoryBuffer offsets = DeviceMemoryBuffer.allocate(16)) {
      header.writeTo(host, 0);
      positions.setLong(0, 0);
      positions.setLong(8, host.getLength());
      data.copyFromHostBuffer(host);
      offsets.copyFromHostBuffer(positions);
      CudfException error = assertThrows(CudfException.class,
          () -> KudoGpuSerializer.assembleFromDeviceRaw(KudoStringOffsetsTest.schema(1), data, offsets));
      assertTrue(error.getMessage().contains("Invalid Kudo section lengths"));
    }
  }

  @Test void gpuMergesWideCharsAboveTwoGiBAfterInputBuffersClose() throws Exception {
    int rows = 2049;
    int bytesPerRow = 1 << 20;
    HostMemoryBuffer data = HostMemoryBuffer.allocate(rows * (long) bytesPerRow, false);
    data.setMemory(0, data.getLength(), (byte) 'w');
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate((rows + 1L) * 8, false);
    for (int row = 0; row <= rows; row++) offsets.setLong(row * 8L, row * (long) bytesPerRow);
    try (HostColumnVector host = new HostColumnVector(DType.STRING, rows, Optional.of(0L),
             data, null, offsets, Collections.emptyList(), DType.INT64);
         ColumnVector device = host.copyToDevice();
         Table input = new Table(device)) {
      DeviceMemoryBuffer[] packed = KudoGpuSerializer.splitAndSerializeToDevice(input, 1024);
      Table output;
      try (DeviceMemoryBuffer payload = packed[0]; DeviceMemoryBuffer positions = packed[1]) {
        assertTrue(payload.getLength() > Integer.MAX_VALUE);
        output = KudoGpuSerializer.assembleFromDeviceRaw(
            KudoStringOffsetsTest.schema(1), payload, positions);
      }
      try (Table restored = output;
           HostColumnVector actual = restored.getColumn(0).copyToHost()) {
        assertEquals(DType.INT64, actual.getOffsetsType());
        assertEquals(rows * (long) bytesPerRow, actual.getEndListOffset(rows - 1));
        assertTrue(actual.getStartListOffset(rows - 1) > Integer.MAX_VALUE);
        byte[] expected = new byte[bytesPerRow];
        Arrays.fill(expected, (byte) 'w');
        for (int row = 0; row < rows; row++) assertArrayEquals(expected, actual.getUTF8(row));
      }
    }
  }
}
