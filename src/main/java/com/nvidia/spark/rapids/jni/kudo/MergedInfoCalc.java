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
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.schema.SimpleSchemaVisitor;
import com.nvidia.spark.rapids.jni.schema.Visitors;

import java.util.*;

import static com.nvidia.spark.rapids.jni.kudo.ColumnOffsetInfo.INVALID_OFFSET;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.*;


/**
 * This class is used to calculate column offsets of merged buffer.
 */
class MergedInfoCalc implements SimpleSchemaVisitor {

    private final KudoTable[] kudoTables;
    // Total data len in gpu, which accounts for 64 byte alignment
    private long totalDataLen;
    private final boolean[] hasNull;
    private final long[] rowCount;
    private final long[] dataLen;
    private final DType[] offsetsTypes;

    // Column offset in gpu device buffer, it has one field for each flattened column
    private final ColumnOffsetInfo[] columnOffsets;
    private int curColIdx = 0;


    MergedInfoCalc(KudoTable[] tables) {
        this.kudoTables = tables;
        this.totalDataLen = 0;
        int columnCount = tables[0].getHeader().getNumColumns();
        this.hasNull = new boolean[columnCount];
        initHasNull();
        this.rowCount = new long[columnCount];
        this.dataLen = new long[columnCount];
        this.offsetsTypes = new DType[columnCount];
        Arrays.fill(offsetsTypes, DType.INT32);
        this.columnOffsets = new ColumnOffsetInfo[columnCount];
    }

    private void doCalc(Schema schema) {
        for (KudoTable kudoTable : kudoTables) {
            Visitors.visitSchema(schema, new SingleTableVisitor(kudoTable));
        }
    }

    public long getTotalDataLen() {
        return totalDataLen;
    }

    ColumnOffsetInfo[] getColumnOffsets() {
        return columnOffsets;
    }

    public KudoTable[] getTables() {
        return kudoTables;
    }

    public long[] getRowCount() {
        return rowCount;
    }

    @Override
    public String toString() {
        return "MergedInfoCalc{" +
                "totalDataLen=" + totalDataLen +
                ", columnOffsets=" + Arrays.toString(columnOffsets) +
                ", hasNull=" + Arrays.toString(hasNull) +
                ", rowCount=" + Arrays.toString(rowCount) +
                ", dataLen=" + Arrays.toString(dataLen) +
                '}';
    }

    static MergedInfoCalc calc(Schema schema, KudoTable[] tables) {
        MergedInfoCalc calc = new MergedInfoCalc(tables);
        calc.doCalc(schema);
        Visitors.visitSchema(schema, calc);
        return calc;
    }

    private void initHasNull() {
        int colNum = kudoTables[0].getHeader().getNumColumns();
        int nullBytesLen = KudoTableHeader.lengthOfHasValidityBuffer(colNum);
        byte[] nullBytes = new byte[nullBytesLen];
        for (KudoTable table : kudoTables) {
            byte[] hasValidityBuffer = table.getHeader().getHasValidityBuffer();
            for (int i =0; i < nullBytesLen; i++) {
                nullBytes[i] = (byte) (nullBytes[i] | hasValidityBuffer[i]);
            }
        }
        for (int i = 0; i < colNum; i++) {
            int pos = i / 8;
            int bit = i % 8;
            hasNull[i] = (nullBytes[pos] & (1 << bit)) != 0;
        }
    }

    @Override
    public void visitTopSchema(Schema schema) {
    }

    @Override
    public void preVisitStruct(Schema structType) {
        long validityOffset = INVALID_OFFSET;
        long validityBufferLen = 0;

        if (hasNull[curColIdx]) {
            validityOffset = totalDataLen;
            validityBufferLen = padFor64byteAlignment(getValidityLengthInBytes(rowCount[curColIdx]));
            totalDataLen += validityBufferLen;
        }

        columnOffsets[curColIdx] = new ColumnOffsetInfo(validityOffset, validityBufferLen,
            INVALID_OFFSET, 0, INVALID_OFFSET, 0);
        curColIdx++;
    }

    @Override
    public void visitStruct(Schema structType) {
        // Noop
    }

    @Override
    public void preVisitList(Schema listType) {
        long validityOffset = INVALID_OFFSET;
        long validityBufferLen = 0;

        if (hasNull[curColIdx]) {
            validityOffset = totalDataLen;
            validityBufferLen = padFor64byteAlignment(getValidityLengthInBytes(rowCount[curColIdx]));
            totalDataLen += validityBufferLen;
        }


        long offsetOffset = INVALID_OFFSET;
        long offsetBufferLen = 0;
        if (rowCount[curColIdx] > 0) {
            offsetOffset = totalDataLen;
            offsetBufferLen = padFor64byteAlignment((long) (rowCount[curColIdx] + 1) * Integer.BYTES);
            totalDataLen += offsetBufferLen;
        }

        columnOffsets[curColIdx] = new ColumnOffsetInfo(validityOffset, validityBufferLen,
                offsetOffset,
                offsetBufferLen,
                INVALID_OFFSET, 0);
        curColIdx++;

    }

    @Override
    public void visitList(Schema listType) {
    }

    @Override
    public void visit(Schema primitiveType) {
        long validityOffset = INVALID_OFFSET;
        long validityBufferLen = 0;

        if (hasNull[curColIdx]) {
            validityOffset = totalDataLen;
            validityBufferLen = padFor64byteAlignment(getValidityLengthInBytes(rowCount[curColIdx]));
            totalDataLen += validityBufferLen;
        }

        long offsetOffset = INVALID_OFFSET;
        long offsetBufferLen = 0;

        long dataOffset = INVALID_OFFSET;
        long dataBufferLen = 0;

        if (rowCount[curColIdx] > 0) {
            if (primitiveType.getType().hasOffsets()) {
                offsetOffset = totalDataLen;
                if (dataLen[curColIdx] > Integer.MAX_VALUE) offsetsTypes[curColIdx] = DType.INT64;
                offsetBufferLen = padFor64byteAlignment(
                    (rowCount[curColIdx] + 1) * offsetsTypes[curColIdx].getSizeInBytes());
                totalDataLen += offsetBufferLen;

                dataOffset = totalDataLen;
                dataBufferLen = padFor64byteAlignment(dataLen[curColIdx]);
            } else {
                dataOffset = totalDataLen;
                dataBufferLen = padFor64byteAlignment(
                    rowCount[curColIdx] * primitiveType.getType().getSizeInBytes());
            }
            totalDataLen += dataBufferLen;
        }

        columnOffsets[curColIdx] = new ColumnOffsetInfo(validityOffset, validityBufferLen,
            offsetOffset,
                offsetBufferLen,
                dataOffset, dataBufferLen, offsetsTypes[curColIdx]);
        curColIdx++;
    }

    private class SingleTableVisitor implements SimpleSchemaVisitor {
        private final KudoTable table;
        private final Deque<SliceInfo> sliceInfos = new ArrayDeque<>(8);
        private int curColIdx = 0;
        private long bufferOffset;

        SingleTableVisitor(KudoTable table) {
            this.table = table;
            this.bufferOffset = table.getHeader().startOffsetOfLong(BufferType.OFFSET);
            sliceInfos.addLast(new SliceInfo(table.getHeader().getOffset(), table.getHeader().getNumRows()));
        }

        @Override
        public void visitTopSchema(Schema schema) {
        }

        @Override
        public void preVisitStruct(Schema structType) {
            if (!DType.INT32.equals(table.getHeader().getOffsetsType(curColIdx))) {
                throw new IllegalArgumentException("STRUCT cannot have wide offsets");
            }
            SliceInfo sliceInfo = sliceInfos.getLast();
            rowCount[curColIdx] = Math.addExact(rowCount[curColIdx], sliceInfo.getRowCount());

            curColIdx++;
        }

        @Override
        public void visitStruct(Schema structType) {
            // Noop
        }

        @Override
        public void preVisitList(Schema listType) {
            if (!DType.INT32.equals(table.getHeader().getOffsetsType(curColIdx))) {
                throw new IllegalArgumentException("LIST offsets must be INT32");
            }
            SliceInfo sliceInfo = sliceInfos.getLast();
            rowCount[curColIdx] = Math.addExact(rowCount[curColIdx], sliceInfo.getRowCount());

            if (sliceInfo.getRowCount() > 0) {
                int startOffset = table.getBuffer().getInt(bufferOffset);
                int endOffset = table.getBuffer().getInt(bufferOffset +
                    (long) sliceInfo.getRowCount() * Integer.BYTES);
                SliceInfo nextSliceInfo = new SliceInfo(startOffset, endOffset - startOffset);
                sliceInfos.addLast(nextSliceInfo);

                bufferOffset += padForHostAlignment(
                    (long) (sliceInfo.getRowCount() + 1) * Integer.BYTES);
            } else {
                sliceInfos.addLast(new SliceInfo(0, 0));
            }

            curColIdx++;
        }

        @Override
        public void visitList(Schema listType) {
            sliceInfos.removeLast();
        }

        @Override
        public void visit(Schema primitiveType) {
            if (!primitiveType.getType().hasOffsets() &&
                !DType.INT32.equals(table.getHeader().getOffsetsType(curColIdx))) {
                throw new IllegalArgumentException("Wide offsets require STRING");
            }
            SliceInfo sliceInfo = sliceInfos.getLast();
            rowCount[curColIdx] = Math.addExact(rowCount[curColIdx], sliceInfo.getRowCount());
            if (primitiveType.getType().hasOffsets()) {
                // String offsets use the width stored in each input table header.
                if (DType.INT64.equals(table.getHeader().getOffsetsType(curColIdx))) {
                    offsetsTypes[curColIdx] = DType.INT64;
                }
                if (sliceInfo.getRowCount() > 0) {
                    DType width = table.getHeader().getOffsetsType(curColIdx);
                    if (DType.INT64.equals(width)) offsetsTypes[curColIdx] = DType.INT64;
                    long startOffset = StringOffsets.read(table.getBuffer(), bufferOffset, width);
                    long endOffset = StringOffsets.read(table.getBuffer(), bufferOffset +
                        (long) sliceInfo.getRowCount() * width.getSizeInBytes(), width);
                    if (startOffset < 0 || endOffset < startOffset) {
                        throw new IllegalArgumentException("Invalid serialized string offsets");
                    }
                    dataLen[curColIdx] = Math.addExact(dataLen[curColIdx], endOffset - startOffset);
                    bufferOffset += padForHostAlignment(
                        (sliceInfo.getRowCount() + 1L) * width.getSizeInBytes());
                }
            }
            // We don't need to update data len for non string primitive type
            curColIdx++;
        }
    }
}
