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
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KudoStringOffsetsTest {
  private HostColumnVector wide() {
    HostMemoryBuffer data = HostMemoryBuffer.allocate(3, false);
    data.setBytes(0, "abc".getBytes(StandardCharsets.UTF_8), 0, 3);
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate(16, false);
    offsets.setLong(0, 0);
    offsets.setLong(8, 3);
    return new HostColumnVector(DType.STRING, 1, Optional.of(0L), data, null, offsets,
        Collections.emptyList(), DType.INT64);
  }

  private Schema schema(int count) {
    Schema.Builder builder = Schema.builder();
    for (int i = 0; i < count; i++) builder.addColumn(DType.STRING, "s" + i);
    return builder.build();
  }

  @Test void int32WireRoundTripPreservesOffsetsAndData() throws Exception {
    try (HostColumnVector host = HostColumnVector.fromStrings("abc", "DE", "fghi")) {
      KudoSerializer serializer = new KudoSerializer(schema(1));
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      serializer.writeToStreamWithMetrics(new HostColumnVector[]{host}, out, 0, 3);
      try (KudoTable restored = KudoTable.from(new ByteArrayInputStream(out.toByteArray())).get()) {
        assertEquals(3, restored.getHeader().getNumRows());
        HostMemoryBuffer buffer = restored.getBuffer();
        long offsetStart = restored.getHeader().getValidityBufferLen();
        for (int i = 0; i < 4; i++) {
          assertEquals(new int[]{0, 3, 5, 9}[i], buffer.getInt(offsetStart + i * 4L));
        }
        byte[] data = new byte[9];
        buffer.getBytes(data, 0, offsetStart + restored.getHeader().getOffsetBufferLen(), 9);
        assertArrayEquals("abcDEfghi".getBytes(StandardCharsets.UTF_8), data);
      }
      assertEquals("DE", host.getJavaString(1));
      assertEquals(1, host.getRefCount());
    }
  }

  @Test void wideHeaderCalculationRejectsInsteadOfMiscounting() {
    try (HostColumnVector host = wide()) {
      KudoTableHeaderCalc calculator = new KudoTableHeaderCalc(0, 1, 1);
      UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
          () -> calculator.visit(host));
      assertTrue(failure.getMessage().contains("INT64"));
      assertEquals("abc", host.getJavaString(0));
    }
  }

  @Test void mixedWidthsRejectBeforeWritingAnyHeaderOrClosingInputs() {
    try (HostColumnVector narrow = HostColumnVector.fromStrings("first");
         HostColumnVector wide = wide()) {
      KudoSerializer serializer = new KudoSerializer(schema(2));
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      RuntimeException failure = assertThrows(RuntimeException.class, () ->
          serializer.writeToStreamWithMetrics(new HostColumnVector[]{narrow, wide}, out, 0, 1));
      assertTrue(failure.getCause() instanceof UnsupportedOperationException);
      assertEquals(0, out.size());
      assertEquals("first", narrow.getJavaString(0));
      assertEquals("abc", wide.getJavaString(0));
      assertEquals(1, narrow.getRefCount());
      assertEquals(1, wide.getRefCount());
    }
  }

  @Test void nestedWideStringRejectsBeforeWriting() {
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate(8, false);
    offsets.setInt(0, 0);
    offsets.setInt(4, 1);
    try (HostColumnVector parent = new HostColumnVector(DType.LIST, 1, Optional.of(0L),
        null, null, offsets, Collections.singletonList(wide()))) {
      Schema.Builder builder = Schema.builder();
      builder.addColumn(DType.LIST, "items").addColumn(DType.STRING, "s");
      KudoSerializer serializer = new KudoSerializer(builder.build());
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      RuntimeException failure = assertThrows(RuntimeException.class, () ->
          serializer.writeToStreamWithMetrics(new HostColumnVector[]{parent}, out, 0, 1));
      assertTrue(failure.getCause() instanceof UnsupportedOperationException);
      assertEquals(0, out.size());
      assertEquals(Collections.singletonList("abc"), parent.getList(0));
    }
  }
}
