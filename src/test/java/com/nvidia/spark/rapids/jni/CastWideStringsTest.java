/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostMemoryBuffer;
import java.math.BigDecimal;
import java.util.Collections;
import java.util.Optional;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CastWideStringsTest {
  private static HostColumnVector wideStrings(String... values) {
    try (HostColumnVector narrow = HostColumnVector.fromStrings(values)) {
      HostMemoryBuffer offsets = HostMemoryBuffer.allocate((values.length + 1L) * 8, false);
      for (int i = 0; i < values.length; i++) {
        offsets.setLong(i * 8L, narrow.getStartListOffset(i));
      }
      offsets.setLong(values.length * 8L, narrow.getEndListOffset(values.length - 1));
      HostMemoryBuffer data = narrow.getData();
      HostMemoryBuffer validity = narrow.getValidity();
      if (data != null) data.incRefCount();
      if (validity != null) validity.incRefCount();
      return new HostColumnVector(DType.STRING, values.length, Optional.of(narrow.getNullCount()),
          data, validity, offsets, Collections.emptyList(), DType.INT64);
    }
  }

  @Test
  void numericValuesAfterTwoGiBOfChars() {
    int paddingRows = 2049;
    int rows = paddingRows + 2;
    long prefixBytes = paddingRows * (1L << 20);
    HostMemoryBuffer data = HostMemoryBuffer.allocate(prefixBytes + 6, false);
    data.setMemory(0, prefixBytes, (byte) 'x');
    data.setBytes(prefixBytes, new byte[]{'2', '6', '4', '8', '8', '3'}, 0, 6);
    HostMemoryBuffer offsets = HostMemoryBuffer.allocate((rows + 1L) * 8, false);
    for (int row = 0; row <= paddingRows; row++) {
      offsets.setLong(row * 8L, row * (1L << 20));
    }
    offsets.setLong((paddingRows + 1L) * 8, prefixBytes + 3);
    offsets.setLong((rows * 8L), prefixBytes + 6);
    try (HostColumnVector host = new HostColumnVector(DType.STRING, rows, Optional.of(0L),
             data, null, offsets, Collections.emptyList(), DType.INT64);
         ColumnVector input = host.copyToDevice();
         ColumnVector integers = CastStrings.toInteger(input, false, DType.INT64);
         HostColumnVector integerValues = integers.copyToHost();
         ColumnVector decimals = CastStrings.toDecimal(input, false, 15, -2);
         HostColumnVector decimalValues = decimals.copyToHost();
         ColumnVector floats = CastStrings.toFloat(input, false, DType.FLOAT64);
         HostColumnVector floatValues = floats.copyToHost()) {
      assertEquals(DType.INT64, input.getOffsetsType());
      assertTrue(prefixBytes > Integer.MAX_VALUE);
      assertEquals(264, integerValues.getLong(paddingRows));
      assertEquals(883, integerValues.getLong(paddingRows + 1));
      assertEquals(new BigDecimal("264.00"), decimalValues.getBigDecimal(paddingRows));
      assertEquals(new BigDecimal("883.00"), decimalValues.getBigDecimal(paddingRows + 1));
      assertEquals(264.0, floatValues.getDouble(paddingRows));
      assertEquals(883.0, floatValues.getDouble(paddingRows + 1));
    }
  }

  @Test
  void integerValuesWithWideOffsets() {
    try (HostColumnVector host = wideStrings("264", "883", null, "", "bad", "-21", " 17 ");
         ColumnVector input = host.copyToDevice()) {
      assertEquals(DType.INT64, input.getOffsetsType());
      for (DType type : new DType[]{DType.INT16, DType.INT32, DType.INT64}) {
        try (ColumnVector output = CastStrings.toInteger(input, false, type);
             ColumnVector longs = output.castTo(DType.INT64);
             HostColumnVector actual = longs.copyToHost()) {
          assertEquals(264, actual.getLong(0));
          assertEquals(883, actual.getLong(1));
          assertTrue(actual.isNull(2));
          assertTrue(actual.isNull(3));
          assertTrue(actual.isNull(4));
          assertEquals(-21, actual.getLong(5));
          assertEquals(17, actual.getLong(6));
        }
      }
    }
  }

  @Test
  void decimalValuesWithWideOffsets() {
    try (HostColumnVector host = wideStrings("264.12", "883.00", null, "", "bad", "-21.34");
         ColumnVector input = host.copyToDevice()) {
      assertEquals(DType.INT64, input.getOffsetsType());
      for (int precision : new int[]{7, 15, 30}) {
        try (ColumnVector output = CastStrings.toDecimal(input, false, precision, -2);
             HostColumnVector actual = output.copyToHost()) {
          assertEquals(new BigDecimal("264.12"), actual.getBigDecimal(0));
          assertEquals(new BigDecimal("883.00"), actual.getBigDecimal(1));
          assertTrue(actual.isNull(2));
          assertTrue(actual.isNull(3));
          assertTrue(actual.isNull(4));
          assertEquals(new BigDecimal("-21.34"), actual.getBigDecimal(5));
        }
      }
    }
  }

  @Test
  void floatValuesWithWideOffsets() {
    try (HostColumnVector host = wideStrings("264.125", "883.00", null, "", "bad", "-21.5");
         ColumnVector input = host.copyToDevice()) {
      assertEquals(DType.INT64, input.getOffsetsType());
      for (DType type : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
        try (ColumnVector output = CastStrings.toFloat(input, false, type);
             ColumnVector doubles = output.castTo(DType.FLOAT64);
             HostColumnVector actual = doubles.copyToHost()) {
          assertEquals(264.125, actual.getDouble(0));
          assertEquals(883.0, actual.getDouble(1));
          assertTrue(actual.isNull(2));
          assertTrue(actual.isNull(3));
          assertTrue(actual.isNull(4));
          assertEquals(-21.5, actual.getDouble(5));
        }
      }
    }
  }

  @Test
  void floatAnsiErrorPreservesWideInput() {
    try (HostColumnVector host = wideStrings("264.125", null, "883.0", "invalid-float");
         ColumnVector input = host.copyToDevice()) {
      CastException error = assertThrows(CastException.class,
          () -> {
            try (ColumnVector ignored = CastStrings.toFloat(input, true, DType.FLOAT64)) {
            }
          });
      assertEquals(3, error.getRowWithError());
      assertEquals("invalid-float", error.getStringWithError());
    }
  }

  @Test
  void integerAnsiErrorPreservesWideInput() {
    try (HostColumnVector host = wideStrings("264", null, "883", "invalid-integer");
         ColumnVector input = host.copyToDevice()) {
      CastException error = assertThrows(CastException.class,
          () -> {
            try (ColumnVector ignored = CastStrings.toInteger(input, true, DType.INT64)) {
            }
          });
      assertEquals(3, error.getRowWithError());
      assertEquals("invalid-integer", error.getStringWithError());
    }
  }

  @Test
  void decimalAnsiErrorPreservesWideInput() {
    try (HostColumnVector host = wideStrings("264.12", null, "883.00", "invalid-decimal");
         ColumnVector input = host.copyToDevice()) {
      CastException error = assertThrows(CastException.class,
          () -> {
            try (ColumnVector ignored = CastStrings.toDecimal(input, true, 15, -2)) {
            }
          });
      assertEquals(3, error.getRowWithError());
      assertEquals("invalid-decimal", error.getStringWithError());
    }
  }
}
