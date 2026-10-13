/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KudoStringOffsetsGpuTest {
  private static void assertStrings(HostColumnVector host, String... values) {
    assertEquals(values.length, host.getRowCount());
    for (int row = 0; row < values.length; row++) {
      if (values[row] == null) assertTrue(host.isNull(row));
      else assertEquals(values[row], host.getJavaString(row));
    }
  }
  @Test void mixedInputWidthsMergeAndRestoreOnGpu() throws Exception {
    KudoSerializer serializer = new KudoSerializer(KudoStringOffsetsTest.schema(1));
    try (HostColumnVector a = KudoStringOffsetsTest.strings(false, "abc", null);
         HostColumnVector b = KudoStringOffsetsTest.strings(true, "", "\u4f60\u597d");
         KudoTable first = KudoStringOffsetsTest.serialize(serializer, 0, 2, a);
         KudoTable second = KudoStringOffsetsTest.serialize(serializer, 0, 2, b);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{first, second});
         Table table = merged.toTable();
         HostColumnVector restored = table.getColumn(0).copyToHost()) {
      assertEquals(DType.INT64, table.getColumn(0).getOffsetsType());
      assertEquals(DType.INT64, restored.getOffsetsType());
      assertStrings(restored, "abc", null, "", "\u4f60\u597d");
    }
  }
  @Test void narrowInputRemainsNarrowThroughGpuRestore() throws Exception {
    KudoSerializer serializer = new KudoSerializer(KudoStringOffsetsTest.schema(1));
    try (HostColumnVector input = KudoStringOffsetsTest.strings(false, "abc", null, "DE");
         KudoTable packed = KudoStringOffsetsTest.serialize(serializer, 0, 3, input);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{packed});
         Table table = merged.toTable();
         HostColumnVector restored = table.getColumn(0).copyToHost()) {
      assertEquals(DType.INT32, table.getColumn(0).getOffsetsType());
      assertStrings(restored, "abc", null, "DE");
    }
  }
  @Test void listStringChildWidthSurvivesGpuRestore() throws Exception {
    Schema.Builder builder = Schema.builder();
    builder.addColumn(DType.LIST, "items").addColumn(DType.STRING, "s");
    KudoSerializer serializer = new KudoSerializer(builder.build());
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate(8, false);
    offsets.setInt(0, 0);
    offsets.setInt(4, 3);
    try (HostColumnVector input = new HostColumnVector(DType.LIST, 1, Optional.of(0L), null,
             null, offsets, Collections.singletonList(KudoStringOffsetsTest.strings(true, "abc", null, "DE")));
         KudoTable packed = KudoStringOffsetsTest.serialize(serializer, 0, 1, input);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{packed, packed});
         Table table = merged.toTable();
         HostColumnVector restored = table.getColumn(0).copyToHost()) {
      assertEquals(DType.INT32, table.getColumn(0).getOffsetsType());
      assertEquals(DType.INT64, restored.getChildColumnView(0).getOffsetsType());
      assertEquals(Arrays.asList("abc", null, "DE"), restored.getList(0));
      assertEquals(Arrays.asList("abc", null, "DE"), restored.getList(1));
    }
  }
  @Test void wideSliceNullAndEmptyStringsSurviveGpuRestore() throws Exception {
    KudoSerializer serializer = new KudoSerializer(KudoStringOffsetsTest.schema(1));
    try (HostColumnVector input = KudoStringOffsetsTest.strings(true, "prefix", "", null, "DE", "suffix");
         KudoTable packed = KudoStringOffsetsTest.serialize(serializer, 1, 3, input);
         KudoHostMergeResult merged = serializer.mergeOnHost(new KudoTable[]{packed});
         Table table = merged.toTable();
         HostColumnVector restored = table.getColumn(0).copyToHost()) {
      assertEquals(DType.INT64, restored.getOffsetsType());
      assertStrings(restored, "", null, "DE");
    }
  }
}
