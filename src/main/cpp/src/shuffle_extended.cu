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

#include "shuffle_split.hpp"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cudf/utilities/traits.hpp>

#include <cstring>
#include <limits>

namespace spark_rapids_jni {
namespace {
constexpr uint32_t narrow_magic = 0x4b554430;
constexpr uint32_t wide_magic   = 0x4b554431;

size_t pad4(size_t value) { return (value + 3) / 4 * 4; }

void put_be(std::vector<uint8_t>& out, size_t at, uint64_t value, int bytes)
{
  for (int i = 0; i < bytes; ++i) { out[at + i] = value >> (8 * (bytes - i - 1)); }
}

uint64_t get_be(uint8_t const* data, size_t at, int bytes)
{
  uint64_t result = 0;
  for (int i = 0; i < bytes; ++i) { result = (result << 8) | data[at + i]; }
  return result;
}

template <typename T>
T scalar_from_device(uint8_t const* data, cuda::stream_ref stream)
{
  T value;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&value, data, sizeof(T), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  return value;
}

int64_t offset_from_device(uint8_t const* data, int width, cuda::stream_ref stream)
{
  return width == 8 ? scalar_from_device<int64_t>(data, stream)
                    : scalar_from_device<int32_t>(data, stream);
}

__device__ int64_t load_offset(uint8_t const* data, int width)
{
  uint64_t result = 0;
  for (int i = 0; i < width; ++i) { result |= uint64_t{data[i]} << (i * 8); }
  return width == 8 ? static_cast<int64_t>(result) : static_cast<int32_t>(result);
}

__global__ void rebase_offsets(uint8_t const* source,
                               uint8_t* destination,
                               int width,
                               cudf::size_type rows,
                               int64_t first)
{
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i <= rows;
       i += int64_t{gridDim.x} * blockDim.x) {
    int64_t const value = rows == 0 ? 0 : load_offset(source + i * width, width) - first;
    if (width == 8) { reinterpret_cast<int64_t*>(destination)[i] = value; }
    else { reinterpret_cast<int32_t*>(destination)[i] = static_cast<int32_t>(value); }
  }
}

__global__ void copy_mask(uint8_t const* source,
                          cudf::bitmask_type* destination,
                          cudf::size_type rows,
                          int bit_shift)
{
  for (int64_t word = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
       word * 32 < rows; word += int64_t{gridDim.x} * blockDim.x) {
    uint32_t value = 0;
    for (int bit = 0; bit < 32 && word * 32 + bit < rows; ++bit) {
      auto const at = word * 32 + bit + bit_shift;
      value |= uint32_t{(source[at / 8] >> (at % 8)) & 1U} << bit;
    }
    destination[word] = value;
  }
}

struct wire_column {
  cudf::column_view view;
  bool validity;
  int width;
  size_t validity_bytes;
  size_t offsets_bytes;
  size_t data_bytes;
};

void flatten(cudf::column_view const& column,
              std::vector<wire_column>& columns,
              shuffle_split_metadata& metadata,
              cuda::stream_ref stream)
{
  auto const type = column.type().id();
  auto const rows = column.size();
  bool const validity = column.nullable() && rows > 0;
  int width = 4;
  size_t offsets = 0;
  size_t bytes = 0;
  if (type == cudf::type_id::STRING) {
    cudf::strings_column_view strings(column);
    if (rows > 0) {
      width = strings.offsets().type().id() == cudf::type_id::INT64 ? 8 : 4;
      offsets = (static_cast<size_t>(rows) + 1) * width;
      bytes = strings.chars_size(stream);
    }
  } else if (type == cudf::type_id::LIST) {
    offsets = rows > 0 ? (static_cast<size_t>(rows) + 1) * 4 : 0;
  } else if (cudf::is_fixed_width(column.type())) {
    bytes = static_cast<size_t>(rows) * cudf::size_of(column.type());
  } else {
    CUDF_EXPECTS(type == cudf::type_id::STRUCT, "Unsupported Kudo GPU column type");
  }
  columns.push_back({column, validity, width, validity ? (static_cast<size_t>(rows) + 7) / 8 : 0,
                     offsets, bytes});
  auto const parameter = type == cudf::type_id::STRUCT ? column.num_children()
    : type == cudf::type_id::LIST ? 1
    : cudf::is_fixed_point(column.type()) ? column.type().scale() : 0;
  metadata.col_info.emplace_back(type, parameter);
  if (type == cudf::type_id::STRUCT) {
    for (auto child = column.child_begin(); child != column.child_end(); ++child) {
      flatten(*child, columns, metadata, stream);
    }
  } else if (type == cudf::type_id::LIST) {
    flatten(cudf::lists_column_view(column).child(), columns, metadata, stream);
  }
}

struct input_column {
  cudf::size_type rows;
  int64_t row_start;
  int width;
  int64_t first;
  int64_t chars;
  bool validity;
  uint8_t const* mask;
  uint8_t const* offsets;
  uint8_t const* data;
};

struct partition_reader {
  shuffle_split_metadata const& metadata;
  uint8_t const* base;
  size_t length;
  std::vector<uint8_t> header;
  size_t validity_at;
  size_t offsets_at;
  size_t data_at;
  size_t validity_end;
  size_t offsets_end;
  size_t data_end;
  bool extended;
  std::vector<input_column> columns;
  cuda::stream_ref stream;

  void check(size_t at, size_t bytes, size_t end)
  {
    CUDF_EXPECTS(at <= end && bytes <= end - at, "Kudo buffer exceeds its declared section");
  }

  size_t parse(size_t index, cudf::size_type rows, int64_t row_start)
  {
    CUDF_EXPECTS(index < metadata.col_info.size(), "Invalid Kudo nested schema");
    auto const& meta = metadata.col_info[index];
    auto const bitmap_bytes = (metadata.col_info.size() + 7) / 8;
    auto const fixed_header = extended ? 40 : 28;
    bool const validity = (header[fixed_header + index / 8] >> (index % 8)) & 1;
    bool const wide = extended &&
      ((header[fixed_header + bitmap_bytes + index / 8] >> (index % 8)) & 1);
    CUDF_EXPECTS(!wide || meta.type == cudf::type_id::STRING,
                 "Only STRING can have wide Kudo offsets");
    int const width = wide ? 8 : 4;
    size_t const mask_bytes = validity && rows > 0 ? (static_cast<size_t>(rows) + row_start % 8 + 7) / 8 : 0;
    check(validity_at, mask_bytes, validity_end);
    auto const mask = base + validity_at;
    validity_at += mask_bytes;
    bool const has_offsets = meta.type == cudf::type_id::STRING || meta.type == cudf::type_id::LIST;
    size_t const offset_bytes = has_offsets && rows > 0 ? (static_cast<size_t>(rows) + 1) * width : 0;
    check(offsets_at, offset_bytes, offsets_end);
    auto const offsets = base + offsets_at;
    offsets_at += offset_bytes;
    int64_t const first = offset_bytes > 0 ? offset_from_device(offsets, width, stream) : 0;
    int64_t const last = offset_bytes > 0
      ? offset_from_device(offsets + int64_t{rows} * width, width, stream) : 0;
    CUDF_EXPECTS(first >= 0 && last >= first, "Invalid Kudo offsets range");
    int64_t const chars = meta.type == cudf::type_id::STRING ? last - first : 0;
    cudf::data_type const dtype{meta.type, cudf::is_fixed_point(cudf::data_type{meta.type}) ? meta.scale() : 0};
    size_t const data_bytes = meta.type == cudf::type_id::STRING ? size_t(chars)
      : cudf::is_fixed_width(dtype) ? static_cast<size_t>(rows) * cudf::size_of(dtype) : 0;
    check(data_at, data_bytes, data_end);
    columns[index] = {rows, row_start, width, first, chars, validity, mask, offsets, base + data_at};
    data_at += data_bytes;
    ++index;
    for (int child = 0; child < meta.num_children(); ++child) {
      int64_t const child_rows = meta.type == cudf::type_id::LIST ? last - first : rows;
      CUDF_EXPECTS(child_rows <= std::numeric_limits<cudf::size_type>::max(),
                   "Kudo LIST child row count exceeds INT32");
      index = parse(index, static_cast<cudf::size_type>(child_rows),
                     meta.type == cudf::type_id::LIST ? first : row_start);
    }
    return index;
  }

  std::pair<size_t, std::unique_ptr<cudf::column>> build(
    size_t index, rmm::device_async_resource_ref mr)
  {
    auto const& meta = metadata.col_info[index];
    auto const& info = columns[index];
    cudf::data_type const dtype{meta.type, cudf::is_fixed_point(cudf::data_type{meta.type}) ? meta.scale() : 0};
    auto mask = cudf::create_null_mask(info.rows,
      info.validity ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED, stream, mr);
    cudf::size_type nulls = 0;
    if (info.validity && info.rows > 0) {
      copy_mask<<<(int64_t{info.rows} + 4095) / 4096, 128, 0, stream.get()>>>(
        info.mask, reinterpret_cast<cudf::bitmask_type*>(mask.data()), info.rows, info.row_start % 8);
      CUDF_CUDA_TRY(cudaGetLastError());
      nulls = cudf::null_count(reinterpret_cast<cudf::bitmask_type const*>(mask.data()),
                               0, info.rows, stream);
    }
    size_t const bytes = meta.type == cudf::type_id::STRING ? size_t(info.chars)
      : cudf::is_fixed_width(dtype) ? static_cast<size_t>(info.rows) * cudf::size_of(dtype) : 0;
    rmm::device_buffer data(bytes, stream, mr);
    if (bytes > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(data.data(), info.data, bytes, cudaMemcpyDeviceToDevice, stream.get()));
    }
    std::vector<std::unique_ptr<cudf::column>> children;
    if (meta.type == cudf::type_id::STRING || meta.type == cudf::type_id::LIST) {
      rmm::device_buffer offsets((static_cast<size_t>(info.rows) + 1) * info.width, stream, mr);
      rebase_offsets<<<(int64_t{info.rows} + 128) / 128, 128, 0, stream.get()>>>(
        info.offsets, static_cast<uint8_t*>(offsets.data()), info.width, info.rows, info.first);
      CUDF_CUDA_TRY(cudaGetLastError());
      children.push_back(std::make_unique<cudf::column>(
        cudf::data_type{info.width == 8 ? cudf::type_id::INT64 : cudf::type_id::INT32},
        info.rows + 1, std::move(offsets),
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr), 0));
    }
    size_t next = index + 1;
    for (int child = 0; child < meta.num_children(); ++child) {
      auto result = build(next, mr);
      next = result.first;
      children.push_back(std::move(result.second));
    }
    return {next, std::make_unique<cudf::column>(dtype, info.rows, std::move(data),
                                                std::move(mask), nulls, std::move(children))};
  }
};

std::pair<size_t, std::unique_ptr<cudf::column>> preserve_width(
  std::unique_ptr<cudf::column> column, size_t index, std::vector<bool> const& wide,
  cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto const type = column->type();
  auto const rows = column->size();
  auto const nulls = column->null_count();
  auto contents = column->release();
  size_t next = index + 1;
  if (type.id() == cudf::type_id::STRING) {
    if (rows > 0 && wide[index] && contents.children[0]->type().id() == cudf::type_id::INT32) {
      contents.children[0] = cudf::cast(contents.children[0]->view(),
                                        cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
  } else if (type.id() == cudf::type_id::STRUCT || type.id() == cudf::type_id::LIST) {
    auto const start = type.id() == cudf::type_id::LIST ? 1 : 0;
    for (size_t child = start; child < contents.children.size(); ++child) {
      auto result = preserve_width(std::move(contents.children[child]), next, wide, stream, mr);
      next = result.first;
      contents.children[child] = std::move(result.second);
    }
  }
  return {next, std::make_unique<cudf::column>(type, rows, std::move(*contents.data),
                                              std::move(*contents.null_mask), nulls,
                                              std::move(contents.children))};
}

std::vector<size_t> read_partition_offsets(cudf::device_span<size_t const> offsets,
                                          size_t bytes, cuda::stream_ref stream)
{
  CUDF_EXPECTS(offsets.size() >= 2, "Kudo GPU assembly requires at least one partition");
  std::vector<size_t> result(offsets.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(result.data(), offsets.data(), offsets.size() * sizeof(size_t),
                                cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  CUDF_EXPECTS(result.front() == 0 && result.back() <= bytes, "Invalid Kudo partition offsets");
  CUDF_EXPECTS(std::is_sorted(result.begin(), result.end()), "Unsorted Kudo partition offsets");
  return result;
}
}  // namespace

shuffle_split_output shuffle_split_extended(cudf::table_view const& input,
                                            std::vector<cudf::size_type> const& splits,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto packed = cudf::contiguous_split(input, splits, stream, mr);
  struct partition {
    std::vector<wire_column> columns;
    std::vector<uint8_t> header;
    size_t validity, offsets, data;
  };
  std::vector<partition> descriptions;
  std::vector<size_t> offsets{0};
  shuffle_split_metadata metadata;
  for (auto const& table : packed) {
    partition current{};
    shuffle_split_metadata schema;
    for (auto const& column : table.table) { flatten(column, current.columns, schema, stream); }
    metadata = std::move(schema);
    size_t const bitmap_bytes = (current.columns.size() + 7) / 8;
    current.header.resize(40 + 2 * bitmap_bytes, 0);
    for (size_t i = 0; i < current.columns.size(); ++i) {
      auto const& col = current.columns[i];
      current.validity += col.validity_bytes;
      current.offsets += col.offsets_bytes;
      current.data += col.data_bytes;
      if (col.validity) { current.header[40 + i / 8] |= 1 << (i % 8); }
      if (col.view.type().id() == cudf::type_id::STRING && col.width == 8) {
        current.header[40 + bitmap_bytes + i / 8] |= 1 << (i % 8);
      }
    }
    current.validity = pad4(current.validity);
    current.offsets = pad4(current.offsets);
    current.data = pad4(current.data);
    put_be(current.header, 0, wide_magic, 4);
    put_be(current.header, 4, 0, 4);
    put_be(current.header, 8, table.table.num_rows(), 4);
    put_be(current.header, 12, current.validity, 8);
    put_be(current.header, 20, current.offsets, 8);
    put_be(current.header, 28, current.validity + current.offsets + current.data, 8);
    put_be(current.header, 36, current.columns.size(), 4);
    offsets.push_back(offsets.back() + current.header.size() + current.validity +
                      current.offsets + current.data);
    descriptions.push_back(std::move(current));
  }
  auto output = std::make_unique<rmm::device_buffer>(offsets.back(), stream, mr);
  auto base = static_cast<uint8_t*>(output->data());
  for (size_t part = 0; part < descriptions.size(); ++part) {
    auto const& desc = descriptions[part];
    auto dest = base + offsets[part];
    CUDF_CUDA_TRY(cudaMemcpyAsync(dest, desc.header.data(), desc.header.size(),
                                  cudaMemcpyHostToDevice, stream.get()));
    size_t validity_at = desc.header.size();
    size_t offsets_at = validity_at + desc.validity;
    size_t data_at = offsets_at + desc.offsets;
    for (auto const& col : desc.columns) {
      auto copy = [&](size_t& at, void const* source, size_t bytes) {
        if (bytes > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(dest + at, source, bytes, cudaMemcpyDeviceToDevice, stream.get()));
          at += bytes;
        }
      };
      copy(validity_at, col.view.null_mask(), col.validity_bytes);
      if (col.offsets_bytes > 0) {
        copy(offsets_at, col.view.child(0).head<uint8_t>(), col.offsets_bytes);
      }
      copy(data_at, col.view.head<uint8_t>(), col.data_bytes);
    }
  }
  rmm::device_uvector<size_t> device_offsets(offsets.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_offsets.data(), offsets.data(), offsets.size() * sizeof(size_t),
                                cudaMemcpyHostToDevice, stream.get()));
  stream.sync();
  return {shuffle_split_result{std::move(output), std::move(device_offsets)}, std::move(metadata)};
}

bool requires_extended_assembly(cudf::device_span<uint8_t const> partitions,
                                cudf::device_span<size_t const> offsets,
                                cuda::stream_ref stream)
{
  if (partitions.size() > size_t(std::numeric_limits<int32_t>::max())) { return true; }
  if (offsets.size() <= 1) { return false; }
  auto const positions = read_partition_offsets(offsets, partitions.size(), stream);
  for (size_t i = 0; i + 1 < positions.size(); ++i) {
    CUDF_EXPECTS(positions[i + 1] - positions[i] >= 4, "Truncated Kudo partition");
    uint8_t magic[4];
    CUDF_CUDA_TRY(cudaMemcpyAsync(magic, partitions.data() + positions[i], 4,
                                  cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    if (get_be(magic, 0, 4) == wide_magic) { return true; }
  }
  return false;
}

shuffle_assemble_result shuffle_assemble_extended(
  shuffle_split_metadata const& metadata, cudf::device_span<uint8_t const> partitions,
  cudf::device_span<size_t const> offsets, cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const positions = read_partition_offsets(offsets, partitions.size(), stream);
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> views;
  std::vector<bool> wide(metadata.col_info.size(), false);
  std::vector<int64_t> column_rows(metadata.col_info.size(), 0);
  int64_t total_rows = 0;
  for (size_t part = 0; part + 1 < positions.size(); ++part) {
    auto const bytes = positions[part + 1] - positions[part];
    auto const base = partitions.data() + positions[part];
    CUDF_EXPECTS(bytes >= 28, "Truncated Kudo partition header");
    std::vector<uint8_t> header(std::min(bytes, size_t{40}));
    CUDF_CUDA_TRY(cudaMemcpyAsync(header.data(), base, header.size(), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    auto const magic = get_be(header.data(), 0, 4);
    CUDF_EXPECTS(magic == narrow_magic || magic == wide_magic, "Invalid Kudo magic");
    bool const extended = magic == wide_magic;
    size_t const fixed = extended ? 40 : 28;
    size_t const bitmap = (metadata.col_info.size() + 7) / 8;
    size_t const header_bytes = fixed + bitmap * (extended ? 2 : 1);
    CUDF_EXPECTS(header_bytes <= bytes, "Truncated Kudo header bitmap");
    header.resize(header_bytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(header.data(), base, header_bytes, cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    auto const rows = get_be(header.data(), 8, 4);
    auto const row_start = get_be(header.data(), 4, 4);
    CUDF_EXPECTS(rows <= uint64_t(std::numeric_limits<cudf::size_type>::max()) &&
                 row_start <= uint64_t(std::numeric_limits<cudf::size_type>::max()),
                 "Kudo row range exceeds INT32");
    CUDF_EXPECTS(get_be(header.data(), extended ? 36 : 24, 4) == metadata.col_info.size(),
                 "Kudo flattened column count does not match schema");
    size_t const validity = get_be(header.data(), 12, extended ? 8 : 4);
    size_t const offset_bytes = get_be(header.data(), extended ? 20 : 16, extended ? 8 : 4);
    size_t const payload = get_be(header.data(), extended ? 28 : 20, extended ? 8 : 4);
    CUDF_EXPECTS(payload <= bytes - header_bytes && validity <= payload &&
                 offset_bytes <= payload - validity, "Invalid Kudo section lengths");
    partition_reader reader{metadata, base, bytes, std::move(header), header_bytes,
      header_bytes + validity, header_bytes + validity + offset_bytes,
      header_bytes + validity, header_bytes + validity + offset_bytes, header_bytes + payload,
      extended, std::vector<input_column>(metadata.col_info.size()), stream};
    size_t index = 0;
    while (index < metadata.col_info.size()) {
      index = reader.parse(index, static_cast<cudf::size_type>(rows), row_start);
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    index = 0;
    while (index < metadata.col_info.size()) {
      auto column = reader.build(index, mr);
      index = column.first;
      columns.push_back(std::move(column.second));
    }
    for (size_t i = 0; i < wide.size(); ++i) {
      column_rows[i] += reader.columns[i].rows;
      CUDF_EXPECTS(column_rows[i] <= std::numeric_limits<cudf::size_type>::max(),
                   "Merged Kudo child row count exceeds INT32");
      wide[i] = wide[i] || (metadata.col_info[i].type == cudf::type_id::STRING &&
                            reader.columns[i].width == 8);
    }
    total_rows += rows;
    CUDF_EXPECTS(total_rows <= std::numeric_limits<cudf::size_type>::max(),
                 "Merged Kudo row count exceeds INT32");
    tables.push_back(std::make_unique<cudf::table>(std::move(columns)));
    views.push_back(tables.back()->view());
  }
  auto merged = tables.size() == 1 ? std::move(tables.front()) : cudf::concatenate(views, stream, mr);
  auto columns = merged->release();
  size_t index = 0;
  for (auto& column : columns) {
    auto rebuilt = preserve_width(std::move(column), index, wide, stream, mr);
    index = rebuilt.first;
    column = std::move(rebuilt.second);
  }
  cudf::table output(std::move(columns));
  auto packed = cudf::pack(output.view(), stream, mr);
  auto view = cudf::unpack(packed);
  shuffle_assemble_result result(std::move(*packed.gpu_data), {});
  for (auto const& column : view) {
    result.column_views.push_back(std::make_unique<cudf::column_view>(column));
  }
  stream.sync();
  return result;
}
}  // namespace spark_rapids_jni
