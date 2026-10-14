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
