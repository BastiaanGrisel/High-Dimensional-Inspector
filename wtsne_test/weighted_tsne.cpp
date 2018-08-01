#include "weighted_tsne.h"

int weighted_tsne::initialise_tsne(std::vector<scalar_type> in_data, int num_dps, int num_dimensions)
{
	try {
		//Input
		data = in_data;
		//data.resize(num_data_points * num_dimensions);
		//std::copy(data.begin(), data.begin() + (num_data_points * num_dimensions), in_data.begin());

		float similarities_comp_time = 0;

		num_data_points = num_dps;
		input_dimensions = num_dimensions;

		hdi::utils::CoutLog log;
		hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type distributions;
		std::vector<scalar_type> probabilities;
		std::vector<int> indices;

		//prob_gen_param._perplexity_multiplier = 20;

		{
			hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);
			prob_gen.computeProbabilityDistributions(data.data(), input_dimensions, num_data_points, distributions, prob_gen_param);
		}

		{
			tSNE.initialize(distributions, &embedding, tSNE_param);
		}

		hdi::utils::secureLogValue(&log, "Similarities computation (sec)", similarities_comp_time);
		return 0;
	}
	catch (std::logic_error& ex) { std::cout << "Logic error: " << ex.what() << std::endl; }
	catch (std::runtime_error& ex) { std::cout << "Runtime error: " << ex.what() << std::endl; }
	catch (...) { std::cout << "An unknown error occurred" << std::endl;; }
	return 1;
}

int weighted_tsne::initialise_tsne(std::wstring data_path, int num_dps, int num_dimensions)
{
	try {
		float data_loading_time = 0;

		//Input
		data.resize(num_dps * num_dimensions);

		{
			hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_loading_time);
			std::ifstream input_file(data_path, std::ios::in | std::ios::binary | std::ios::ate);
			if (int(input_file.tellg()) != int(sizeof(scalar_type) * num_dimensions * num_dps)) {
				std::cout << "Input file size doesn't agree with input parameters!" << std::endl;
				return 1;
			}
			input_file.seekg(0, std::ios::beg);
			input_file.read(reinterpret_cast<char*>(data.data()), sizeof(scalar_type) * num_dimensions * num_dps);
			input_file.close();
		}

		hdi::utils::CoutLog log;
		hdi::utils::secureLogValue(&log, "Data loading (sec)", data_loading_time);

		initialise_tsne(data, num_dps, num_dimensions);

		hdi::utils::secureLogValue(&log, "Data loading (sec)", data_loading_time);

		return 0;
	}
	catch (std::logic_error& ex) { std::cout << "Logic error: " << ex.what() << std::endl; }
	catch (std::runtime_error& ex) { std::cout << "Runtime error: " << ex.what() << std::endl; }
	catch (...) { std::cout << "An unknown error occurred" << std::endl;; }
	return 1;
}

void weighted_tsne::do_iteration() {
	tSNE.doAnIteration();
	//hdi::utils::CoutLog log;
	//hdi::utils::secureLogValue(&log, "Iter", iter, verbose);
}

std::vector<int> intersection(std::vector<int> &v1, std::vector<int> &v2)
{
	std::vector<int> v3;

	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());

	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v3));

	return v3;
}

void weighted_tsne::compute_neighbours(std::vector<float> data, int N, int d, int k, std::vector<int> &res) {

	std::vector<weighted_tsne::scalar_type> distances_squared;
	hdi::dr::HDJointProbabilityGenerator<weighted_tsne::scalar_type>::Parameters temp_prob_gen_param;
	//temp_prob_gen_param._perplexity = k;
	//temp_prob_gen_param._perplexity_multiplier = 1;

	// computeHighDimensionalDistances includes the point itself as its nearest neighbour (returns k+1 neighbours)
	//prob_gen.computeHighDimensionalDistances(data.data(), d, N, distances_squared, res, temp_prob_gen_param);
}

void weighted_tsne::compute_weight_falloff(std::vector<float> in_data, int N, int d, std::vector<int> selected_indices, int k, std::vector<float> &weights_falloff) {
	std::vector<int> nn;
	compute_neighbours(in_data, N, d, k, nn);// TODO: inly calculate neighbours of selectedIndices, not of all points

	std::vector<int> min_nn(N, INT_MAX);

	// For each point, save the k-nn distance to the closest selected point. Save the distance to the closest selected point.
	for (int selectedIndex : selected_indices) {
		for (int j = 0; j <= k; j++) {
			int idx = selectedIndex * (k + 1); // Index of the first nearest neighbour of selectedIndex in nn (which is selectedIndex itself)
			int nn_idx = nn[idx + j]; // Index of the j-th nearest neighbour of selectedIndex in min_nn
			min_nn[nn_idx] = std::min(min_nn[nn_idx], j);
			//min_nn[nn_idx] += j;
		}
	}

	weights_falloff.resize(N, 0);

	// Calculate weights from min_nn
	for (int i = 0; i < N; i++) {
		if (min_nn[i] != INT_MAX) {
			weights_falloff[i] = (k - min_nn[i]) / (float)k;
		}
	}
}

int weighted_tsne::read_bin(std::wstring file_path, int N, int d, std::vector<weighted_tsne::scalar_type> &out) {
	out.resize(N * d);

	std::ifstream input_file(file_path, std::ios::in | std::ios::binary | std::ios::ate);

	if (int(input_file.tellg()) != int(sizeof(weighted_tsne::scalar_type) * N * d)) {
		std::cout << "Input file size doesn't agree with input parameters!" << std::endl;
		return 1;
	}

	input_file.seekg(0, std::ios::beg);
	input_file.read(reinterpret_cast<char*>(out.data()), sizeof(weighted_tsne::scalar_type) * N * d);
	input_file.close();

	return 0;
}

int weighted_tsne::read_csv(std::wstring file_path, int N, int d, std::vector<weighted_tsne::scalar_type> &out) {
	out.resize(N * d);

	std::ifstream in_file(file_path, std::ios::in);
	std::string line;

	int i = 0;

	while (std::getline(in_file, line)) {
		std::stringstream          line_stream(line);
		std::string                cell;

		while (std::getline(line_stream, cell, ',')) {
			float v = std::atof(cell.c_str());
			out[i] = v;
			i++;
		}
	}

	return 0;
}

void weighted_tsne::write_csv(std::vector<float> data, int N, int output_dims, std::string filename) {

	std::ofstream out_file2(filename);

	for (int i = 0; i < N; i++) {
		std::string line = "";

		for (int d = 0; d < output_dims; d++) {
			int index = i * output_dims + d;
			line += std::to_string(data[index]);

			if (d < (output_dims - 1)) {
				line += ",";
			}
		}

		out_file2 << line << std::endl;
	}

	out_file2.close();
}


void weighted_tsne::lerp(std::vector<float> from, std::vector<float> to, std::vector<float> &res, float alpha) {
	res.resize(from.size());

	for (int i = 0; i < from.size(); i++) {
		res[i] = (1 - alpha) * from[i] + alpha * to[i];
	}
}
