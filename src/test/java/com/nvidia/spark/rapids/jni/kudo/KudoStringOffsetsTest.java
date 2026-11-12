/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KudoStringOffsetsTest {
  static HostColumnVector strings(boolean wide, String... values) {
    long bytes = 0;
    long nulls = 0;
    for (String value : values) {
      if (value == null) nulls++;
      else bytes += value.getBytes(StandardCharsets.UTF_8).length;
    }
    HostMemoryBuffer data = HostMemoryBuffer.allocate(Math.max(bytes, 1), false);
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate((values.length + 1L) * (wide ? 8 : 4), false);
    HostMemoryBuffer valid = nulls == 0 ? null : HostMemoryBuffer.allocate(64, false);
    if (valid != null) valid.setMemory(0, 64, (byte) 0);
    long at = 0;
    for (int row = 0; row <= values.length; row++) {
      if (wide) offsets.setLong(row * 8L, at);
      else offsets.setInt(row * 4L, Math.toIntExact(at));
      if (row < values.length && values[row] != null) {
        byte[] value = values[row].getBytes(StandardCharsets.UTF_8);
        data.setBytes(at, value, 0, value.length);
        at += value.length;
        if (valid != null) valid.setByte(row / 8L, (byte) (valid.getByte(row / 8L) | (1 << (row % 8))));
      }
    }
    return new HostColumnVector(DType.STRING, values.length, Optional.of(nulls), data, valid,
        offsets, Collections.emptyList(), wide ? DType.INT64 : DType.INT32);
  }
  static Schema schema(int count) {
    Schema.Builder builder = Schema.builder();
    for (int i = 0; i < count; i++) builder.addColumn(DType.STRING, "s" + i);
    return builder.build();
  }
  static KudoTable serialize(KudoSerializer serializer, int offset, int rows,
                             HostColumnVector... columns) throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    serializer.writeToStreamWithMetrics(columns, out, offset, rows);
    return KudoTable.from(new ByteArrayInputStream(out.toByteArray())).get();
  }
  private static void assertMerged(KudoHostMergeResult result, int column, DType width,
                                    String... expected) {
    ColumnViewInfo info = result.getColumnInfoList()[column];
    ColumnOffsetInfo offsets = info.getOffsetInfo();
    HostMemoryBuffer buffer = result.getHostBuf();
    try (HostColumnVector restored = new HostColumnVector(DType.STRING, info.getRowCount(),
        Optional.of(info.getNullCount()), offsets.getData() == -1 ? null :
            buffer.slice(offsets.getData(), offsets.getDataBufferLen()),
        offsets.getValidity() == -1 ? null : buffer.slice(offsets.getValidity(), offsets.getValidityBufferLen()),
        offsets.getOffset() == -1 ? null : buffer.slice(offsets.getOffset(), offsets.getOffsetBufferLen()),
        Collections.emptyList(), offsets.getOffsetsType())) {
      assertEquals(width, restored.getOffsetsType());
      assertEquals(expected.length, restored.getRowCount());
      for (int i = 0; i < expected.length; i++) {
        if (expected[i] == null) assertTrue(restored.isNull(i));
        else assertEquals(expected[i], restored.getJavaString(i));
      }
    }
  }

  @Test void int32WireAndMergeKeepInt32() throws Exception {
    KudoSerializer serializer = new KudoSerializer(schema(1));
    try (HostColumnVector host = HostColumnVector.fromStrings("abc", null, "", "DE");
         KudoTable table = serialize(serializer, 0, 4, host);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table, table})) {
      assertEquals(DType.INT32, table.getHeader().getOffsetsType(0));
      assertEquals(KudoTableHeader.getSerializedSize(1), table.getHeader().getSerializedSize());
      assertMerged(merged, 0, DType.INT32, "abc", null, "", "DE", "abc", null, "", "DE");
    }
  }
  @Test void wideWireAndMergePreserveValues() throws Exception {
    KudoSerializer serializer = new KudoSerializer(schema(1));
    try (HostColumnVector host = strings(true, "abc", null, "", "\u4f60\u597d");
         KudoTable table = serialize(serializer, 0, 4, host);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table})) {
      assertEquals(DType.INT64, table.getHeader().getOffsetsType(0));
      assertEquals(40, table.getHeader().getOffsetBufferLength());
      assertMerged(merged, 0, DType.INT64, "abc", null, "", "\u4f60\u597d");
    }
  }
  @Test void mixedWidthTablesMergeIntoOneWideColumn() throws Exception {
    KudoSerializer serializer = new KudoSerializer(schema(1));
    try (HostColumnVector narrow = strings(false, "first", null);
         HostColumnVector wide = strings(true, "second", "");
         KudoTable a = serialize(serializer, 0, 2, narrow);
         KudoTable b = serialize(serializer, 0, 2, wide);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{a, b})) {
      assertMerged(merged, 0, DType.INT64, "first", null, "second", "");
      assertEquals(1, narrow.getRefCount());
      assertEquals(1, wide.getRefCount());
    }
  }
  @Test void mixedColumnsKeepTheirIndividualWidths() throws Exception {
    KudoSerializer serializer = new KudoSerializer(schema(2));
    try (HostColumnVector narrow = strings(false, "first");
         HostColumnVector wide = strings(true, "second");
         KudoTable table = serialize(serializer, 0, 1, narrow, wide);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table})) {
      assertEquals(DType.INT32, table.getHeader().getOffsetsType(0));
      assertEquals(DType.INT64, table.getHeader().getOffsetsType(1));
      assertMerged(merged, 0, DType.INT32, "first");
      assertMerged(merged, 1, DType.INT64, "second");
    }
  }
  @Test void wideSliceRebasesOffsetsAndValidity() throws Exception {
    KudoSerializer serializer = new KudoSerializer(schema(1));
    try (HostColumnVector wide = strings(true, "prefix", "abc", null, "", "DE", "suffix");
         KudoTable table = serialize(serializer, 1, 4, wide);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table, table})) {
      assertMerged(merged, 0, DType.INT64, "abc", null, "", "DE", "abc", null, "", "DE");
    }
  }
  @Test void nestedListKeepsNarrowParentAndWideChild() throws Exception {
    Schema.Builder builder = Schema.builder();
    builder.addColumn(DType.LIST, "items").addColumn(DType.STRING, "s");
    KudoSerializer serializer = new KudoSerializer(builder.build());
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate(8, false);
    offsets.setInt(0, 0);
    offsets.setInt(4, 3);
    try (HostColumnVector parent = new HostColumnVector(DType.LIST, 1, Optional.of(0L),
             null, null, offsets, Collections.singletonList(strings(true, "abc", null, "DE")));
         KudoTable table = serialize(serializer, 0, 1, parent);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table, table})) {
      assertEquals(DType.INT32, table.getHeader().getOffsetsType(0));
      assertEquals(DType.INT64, table.getHeader().getOffsetsType(1));
      ColumnOffsetInfo parentInfo = merged.getColumnInfoList()[0].getOffsetInfo();
      assertEquals(DType.INT32, parentInfo.getOffsetsType());
      assertEquals(0, merged.getHostBuf().getInt(parentInfo.getOffset()));
      assertEquals(3, merged.getHostBuf().getInt(parentInfo.getOffset() + 4));
      assertEquals(6, merged.getHostBuf().getInt(parentInfo.getOffset() + 8));
      assertMerged(merged, 1, DType.INT64, "abc", null, "DE", "abc", null, "DE");
    }
  }
  @Test void sourceBytePositionBeyondInt32IsRebasedWithoutTruncation() throws Exception {
    long start = (1L << 31) + 17;
    HostMemoryBuffer data = HostMemoryBuffer.allocate(start + 3, false);
    data.setBytes(start, "XYZ".getBytes(StandardCharsets.UTF_8), 0, 3);
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate(16, false);
    offsets.setLong(0, start);
    offsets.setLong(8, start + 3);
    KudoSerializer serializer = new KudoSerializer(schema(1));
    try (HostColumnVector host = new HostColumnVector(DType.STRING, 1, Optional.of(0L), data,
             null, offsets, Collections.emptyList(), DType.INT64);
         KudoTable table = serialize(serializer, 0, 1, host);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table})) {
      assertMerged(merged, 0, DType.INT64, "XYZ");
    }
  }
  @Test void wideAllNullColumnsRoundTrip() throws Exception {
    KudoSerializer serializer = new KudoSerializer(schema(1));
    try (HostColumnVector nulls = strings(true, null, null);
         KudoTable table = serialize(serializer, 0, 2, nulls);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table})) {
      assertMerged(merged, 0, DType.INT64, null, null);
    }
  }
  @Test void zeroRootRowsRetainExistingRejectionBeforeWriting() {
    KudoSerializer serializer = new KudoSerializer(schema(1));
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (HostColumnVector empty = strings(true)) {
      assertThrows(IllegalStateException.class, () ->
          serializer.writeToStreamWithMetrics(new HostColumnVector[]{empty}, out, 0, 0));
      assertEquals(0, out.size());
      assertEquals(1, empty.getRefCount());
    }
  }
  @Test void nestedEmptyWideChildRoundTrip() throws Exception {
    Schema.Builder builder = Schema.builder();
    builder.addColumn(DType.LIST, "items").addColumn(DType.STRING, "s");
    KudoSerializer serializer = new KudoSerializer(builder.build());
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate(8, false);
    offsets.setInt(0, 0);
    offsets.setInt(4, 0);
    try (HostColumnVector parent = new HostColumnVector(DType.LIST, 1, Optional.of(0L),
             null, null, offsets, Collections.singletonList(strings(true)));
         KudoTable table = serialize(serializer, 0, 1, parent);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{table, table})) {
      ColumnOffsetInfo list = merged.getColumnInfoList()[0].getOffsetInfo();
      for (int row = 0; row < 3; row++) assertEquals(0, merged.getHostBuf().getInt(list.getOffset() + row * 4L));
      assertMerged(merged, 1, DType.INT64);
    }
  }
  @Test void oldHeaderEncodingRemainsByteForByte() throws Exception {
    KudoTableHeader header = new KudoTableHeader(0, 1, 0, 8, 12, 1, new byte[]{0});
    ByteArrayOutputStream actual = new ByteArrayOutputStream();
    header.writeTo(new ByteArrayOutputStreamWriter(actual));
    ByteArrayOutputStream expected = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(expected);
    for (int value : new int[]{0x4B554430, 0, 1, 0, 8, 12, 1}) out.writeInt(value);
    out.writeByte(0);
    assertArrayEquals(expected.toByteArray(), actual.toByteArray());
  }
  @Test void wideHeaderLengthsRoundTripWithoutAllocatingPayload() throws Exception {
    long bytes = (1L << 31) + 17;
    KudoTableHeader header = new KudoTableHeader(0, 3, 0L, 32L, bytes, 1,
        new byte[]{0}, new byte[]{1}, true);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    header.writeTo(new ByteArrayOutputStreamWriter(out));
    assertEquals(42, out.size());
    try (HostMemoryBuffer nativeHeader = HostMemoryBuffer.allocate(42, false)) {
      assertEquals(42, header.writeTo(nativeHeader, 0));
      byte[] nativeBytes = new byte[42];
      nativeHeader.getBytes(nativeBytes, 0, 0, 42);
      assertArrayEquals(out.toByteArray(), nativeBytes);
    }
    KudoTableHeader restored = KudoTableHeader.readFrom(new DataInputStream(
        new ByteArrayInputStream(out.toByteArray()))).get();
    assertEquals(bytes, restored.getTotalDataLength());
    assertEquals(DType.INT64, restored.getOffsetsType(0));
    assertThrows(ArithmeticException.class, restored::getTotalDataLen);
  }
  @Test void byteArrayWriterRejectsOversizedReservationBeforeWriting() {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    assertThrows(ArithmeticException.class, () -> new ByteArrayOutputStreamWriter(out).reserve(1L << 31));
    assertEquals(0, out.size());
  }
}
