/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Compatibility adaptive hash-grid wrapper used by the hash-grid collision
 * profiler. The trainable encoding path delegates to the stock HashGrid while
 * exposing an optional per-brick layout for profiling adaptive slot budgets.
 */
#pragma once

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tcnn {

struct AdaptiveHashGridMetadata {
	uint32_t n_levels = 0;
	uint32_t brick_resolution = 0;
	uint32_t n_bricks_per_level = 0;
	uint32_t total_entries = 0;
};

class AdaptiveHashGridEncodingBase {
public:
	virtual ~AdaptiveHashGridEncodingBase() = default;

	virtual AdaptiveHashGridMetadata adaptive_metadata() const = 0;
	virtual const uint32_t* adaptive_brick_offsets_gpu() const = 0;
	virtual const uint32_t* adaptive_brick_log2_sizes_gpu() const = 0;
	virtual uint32_t adaptive_total_entries() const = 0;
};

__host__ __device__ inline uint32_t adaptive_hashgrid_brick_index(
	const AdaptiveHashGridMetadata& meta,
	uint32_t level,
	uint32_t resolution,
	const uvec<3>& local_pos
) {
	if (meta.brick_resolution == 0 || meta.n_bricks_per_level == 0) {
		return 0;
	}

	const uint32_t br = meta.brick_resolution;
	const uint32_t denom = resolution + 1;
	uvec<3> brick_pos;
	TCNN_PRAGMA_UNROLL
	for (uint32_t dim = 0; dim < 3; ++dim) {
		brick_pos[dim] = min((uint32_t)(((uint64_t)local_pos[dim] * br) / denom), br - 1);
	}

	return level * meta.n_bricks_per_level + (brick_pos[2] * br + brick_pos[1]) * br + brick_pos[0];
}

__host__ __device__ inline uint32_t adaptive_hashgrid_entry(
	const AdaptiveHashGridMetadata& meta,
	const uint32_t* brick_offsets,
	const uint32_t* brick_log2_sizes,
	uint32_t level,
	uint32_t resolution,
	const uvec<3>& local_pos
) {
	if (!brick_offsets || !brick_log2_sizes || meta.total_entries == 0) {
		return 0;
	}

	const uint32_t brick = adaptive_hashgrid_brick_index(meta, level, resolution, local_pos);
	const uint32_t local_log2_size = brick_log2_sizes[brick];
	const uint32_t local_size = 1u << min(local_log2_size, 31u);
	const uint32_t local_hash = grid_index<3, HashType::CoherentPrime>(GridType::Hash, local_size, resolution, local_pos);
	return brick_offsets[brick] + local_hash;
}

template <typename T, uint32_t N_POS_DIMS = 3, uint32_t N_FEATURES_PER_LEVEL = 2, HashType HASH_TYPE = HashType::CoherentPrime>
class AdaptiveHashGridEncodingTemplated :
	public GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>,
	public AdaptiveHashGridEncodingBase {
public:
	using Base = GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>;

	AdaptiveHashGridEncodingTemplated(
		uint32_t n_features,
		uint32_t log2_hashmap_size,
		uint32_t base_resolution,
		float per_level_scale,
		bool stochastic_interpolation,
		InterpolationType interpolation_type,
		GridType grid_type,
		bool fixed_point_pos,
		const json& encoding
	) :
	Base{
		n_features,
		log2_hashmap_size,
		base_resolution,
		per_level_scale,
		stochastic_interpolation,
		interpolation_type,
		grid_type,
		fixed_point_pos
	} {
		if (N_POS_DIMS != 3) {
			throw std::runtime_error{"AdaptiveHashGrid: only 3D inputs are supported."};
		}
		if (grid_type != GridType::Hash) {
			throw std::runtime_error{"AdaptiveHashGrid: only Hash grid type is supported."};
		}

		initialize_adaptive_layout(encoding);
	}

	AdaptiveHashGridMetadata adaptive_metadata() const override {
		return m_adaptive_meta;
	}

	const uint32_t* adaptive_brick_offsets_gpu() const override {
		return m_adaptive_brick_offsets.data();
	}

	const uint32_t* adaptive_brick_log2_sizes_gpu() const override {
		return m_adaptive_brick_log2_sizes.data();
	}

	uint32_t adaptive_total_entries() const override {
		return m_adaptive_meta.total_entries;
	}

	json hyperparams() const override {
		json result = Base::hyperparams();
		result["otype"] = "AdaptiveHashGrid";
		result["brick_resolution"] = m_adaptive_meta.brick_resolution;
		result["adaptive_total_entries"] = m_adaptive_meta.total_entries;
		return result;
	}

private:
	void initialize_adaptive_layout(const json& encoding) {
		const uint32_t n_levels = this->output_width() / N_FEATURES_PER_LEVEL;
		const uint32_t brick_resolution = encoding.value("brick_resolution", 0u);

		if (!encoding.contains("layout")) {
			m_adaptive_meta = {n_levels, 0u, 0u, 0u};
			return;
		}

		const json& layout = encoding["layout"];
		const uint32_t layout_brick_resolution = layout.value("brick_resolution", brick_resolution);
		if (layout_brick_resolution == 0) {
			throw std::runtime_error{"AdaptiveHashGrid: layout brick_resolution must be positive."};
		}

		const uint32_t n_bricks_per_level = layout_brick_resolution * layout_brick_resolution * layout_brick_resolution;
		const uint32_t layout_levels = layout.value("n_levels", n_levels);
		if (layout_levels != n_levels) {
			throw std::runtime_error{fmt::format("AdaptiveHashGrid: layout has {} levels, but encoding has {}.", layout_levels, n_levels)};
		}
		if (!layout.contains("brick_log2_sizes")) {
			throw std::runtime_error{"AdaptiveHashGrid: layout must contain brick_log2_sizes."};
		}

		const json& per_level_logs = layout["brick_log2_sizes"];
		if (per_level_logs.size() != n_levels) {
			throw std::runtime_error{fmt::format("AdaptiveHashGrid: expected {} levels of brick_log2_sizes, got {}.", n_levels, per_level_logs.size())};
		}

		std::vector<uint32_t> offsets;
		std::vector<uint32_t> log2_sizes;
		offsets.reserve((size_t)n_levels * n_bricks_per_level);
		log2_sizes.reserve((size_t)n_levels * n_bricks_per_level);

		uint64_t total_entries = 0;
		for (uint32_t level = 0; level < n_levels; ++level) {
			const json& level_logs = per_level_logs[level];
			if (level_logs.size() != n_bricks_per_level) {
				throw std::runtime_error{fmt::format(
					"AdaptiveHashGrid: level {} has {} brick sizes, expected {}.",
					level,
					level_logs.size(),
					n_bricks_per_level
				)};
			}

			for (const auto& log2_size_json : level_logs) {
				const uint32_t log2_size = log2_size_json.get<uint32_t>();
				if (log2_size >= 32) {
					throw std::runtime_error{"AdaptiveHashGrid: brick log2 size must be less than 32."};
				}
				if (total_entries > std::numeric_limits<uint32_t>::max()) {
					throw std::runtime_error{"AdaptiveHashGrid: layout exceeds uint32_t address space."};
				}

				offsets.push_back((uint32_t)total_entries);
				log2_sizes.push_back(log2_size);
				total_entries += 1ull << log2_size;
			}
		}

		if (total_entries > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error{"AdaptiveHashGrid: layout exceeds uint32_t address space."};
		}

		m_adaptive_meta = {
			n_levels,
			layout_brick_resolution,
			n_bricks_per_level,
			(uint32_t)total_entries,
		};
		m_adaptive_brick_offsets.resize_and_copy_from_host(offsets);
		m_adaptive_brick_log2_sizes.resize_and_copy_from_host(log2_sizes);
	}

	AdaptiveHashGridMetadata m_adaptive_meta;
	GPUMemory<uint32_t> m_adaptive_brick_offsets;
	GPUMemory<uint32_t> m_adaptive_brick_log2_sizes;
};

template <typename T, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
MultiLevelEncoding<T>* create_adaptive_hashgrid_encoding_templated_2(uint32_t n_dims_to_encode, const json& encoding) {
	if (n_dims_to_encode != 3) {
		throw std::runtime_error{"AdaptiveHashGrid: number of input dims must be 3."};
	}

	const uint32_t log2_hashmap_size = encoding.value("log2_hashmap_size", 19u);
	uint32_t n_features;
	if (encoding.contains("n_features") || encoding.contains("n_grid_features")) {
		n_features = encoding.contains("n_features") ? encoding["n_features"] : encoding["n_grid_features"];
		if (encoding.contains("n_levels")) {
			throw std::runtime_error{"AdaptiveHashGrid: may not specify n_features and n_levels simultaneously (one determines the other)"};
		}
	} else {
		n_features = N_FEATURES_PER_LEVEL * encoding.value("n_levels", 16u);
	}

	const GridType grid_type = string_to_grid_type(encoding.value("type", std::string{"Hash"}));
	const uint32_t base_resolution = encoding.value("base_resolution", 16u);

	return new AdaptiveHashGridEncodingTemplated<T, 3, N_FEATURES_PER_LEVEL, HASH_TYPE>{
		n_features,
		log2_hashmap_size,
		base_resolution,
		encoding.value("per_level_scale", 2.0f),
		encoding.value("stochastic_interpolation", false),
		string_to_interpolation_type(encoding.value("interpolation", std::string{"Linear"})),
		grid_type,
		encoding.value("fixed_point_pos", false),
		encoding
	};
}

template <typename T, HashType HASH_TYPE>
MultiLevelEncoding<T>* create_adaptive_hashgrid_encoding_templated_1(uint32_t n_dims_to_encode, const json& encoding) {
	const uint32_t n_features_per_level = encoding.value("n_features_per_level", 2u);
	switch (n_features_per_level) {
		case 1: return create_adaptive_hashgrid_encoding_templated_2<T, 1, HASH_TYPE>(n_dims_to_encode, encoding);
		case 2: return create_adaptive_hashgrid_encoding_templated_2<T, 2, HASH_TYPE>(n_dims_to_encode, encoding);
		case 4: return create_adaptive_hashgrid_encoding_templated_2<T, 4, HASH_TYPE>(n_dims_to_encode, encoding);
		case 8: return create_adaptive_hashgrid_encoding_templated_2<T, 8, HASH_TYPE>(n_dims_to_encode, encoding);
		default: throw std::runtime_error{"AdaptiveHashGrid: n_features_per_level must be 1, 2, 4, or 8."};
	}
}

template <typename T>
MultiLevelEncoding<T>* create_adaptive_hashgrid_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	const HashType hash_type = string_to_hash_type(encoding.value("hash", std::string{"CoherentPrime"}));
	switch (hash_type) {
		case HashType::CoherentPrime: return create_adaptive_hashgrid_encoding_templated_1<T, HashType::CoherentPrime>(n_dims_to_encode, encoding);
		default: throw std::runtime_error{"AdaptiveHashGrid: only CoherentPrime hash support is compiled."};
	}
}

} // namespace tcnn
