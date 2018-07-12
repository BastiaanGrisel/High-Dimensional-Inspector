#include "weighted_tsne.h"

int weighted_tsne::initialise_tsne(std::vector<scalar_type> in_data, int num_data_points, int num_dimensions)
{
	try {
		//Input
		data = in_data;
		//data.resize(num_data_points * num_dimensions);
		//std::copy(data.begin(), data.begin() + (num_data_points * num_dimensions), in_data.begin());

		float similarities_comp_time = 0;

		hdi::utils::CoutLog log;
		hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type distributions;

		{
			hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);
			prob_gen.computeProbabilityDistributions(data.data(), num_dimensions, num_data_points, distributions, prob_gen_param);
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

int weighted_tsne::initialise_tsne(std::wstring data_path, int num_data_points, int num_dimensions)
{
	try {
		float data_loading_time = 0;

		//Input
		data.resize(num_data_points * num_dimensions);

		{
			hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_loading_time);
			std::ifstream input_file(data_path, std::ios::in | std::ios::binary | std::ios::ate);
			if (int(input_file.tellg()) != int(sizeof(scalar_type) * num_dimensions * num_data_points)) {
				std::cout << "Input file size doesn't agree with input parameters!" << std::endl;
				return 1;
			}
			input_file.seekg(0, std::ios::beg);
			input_file.read(reinterpret_cast<char*>(data.data()), sizeof(scalar_type) * num_dimensions * num_data_points);
			input_file.close();
		}

		hdi::utils::CoutLog log;
		hdi::utils::secureLogValue(&log, "Data loading (sec)", data_loading_time);

		initialise_tsne(data, num_data_points, num_dimensions);

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


float weighted_tsne::jaccard_similarity(std::vector<int> A, std::vector<int> B) {

	float sizeA = A.size();
	float sizeB = B.size();
	float sizeIntersection = intersection(A, B).size();

	return sizeIntersection / (sizeA + sizeB - sizeIntersection);
}

void weighted_tsne::calculate_percentage_error(std::vector<int> &NN1, std::vector<int> &NN2, std::vector<scalar_type> &errors, int N, int d, int k) {
	assert(k <= d);
	errors.resize(N, 0);

	// Calculate seq error for every data point
	for (int i = 0; i < N; i++)
	{
		// Extract the neighbourhood list from the list of neighbours (and skip the first one)
		int start_index = i * d + 1;
		int end_index = start_index + k;

		// Store the neighbourhoods in a new array for convenience
		std::vector<int> NN1_i(NN1.begin() + start_index, NN1.begin() + end_index); // +1 to skip itself (which is closest neighbour)
		std::vector<int> NN2_i(NN2.begin() + start_index, NN2.begin() + end_index);

		errors[i] = (intersection(NN1_i, NN2_i).size() / (float) k);
	}
}

void weighted_tsne::calculate_set_error(std::vector<int> &NN1, std::vector<int> &NN2, std::vector<scalar_type> &errors, int N, int d, int k) {
	assert(k <= d);
	errors.resize(N, 0);

	// Calculate seq error for every data point
	for (int i = 0; i < N; i++)
	{
		// Extract the neighbourhood list from the list of neighbours (and skip the first one)
		int start_index = i * d + 1;
		int end_index = start_index + k;

		// Store the neighbourhoods in a new array for convenience
		std::vector<int> NN1_i(NN1.begin() + start_index, NN1.begin() + end_index); // +1 to skip itself (which is closest neighbour)
		std::vector<int> NN2_i(NN2.begin() + start_index, NN2.begin() + end_index);

		errors[i] = 1 - jaccard_similarity(NN1_i, NN2_i);
	}
}

int weighted_tsne::indexOf(std::vector<int> in, int value) {
	for (int i = 0; i < in.size(); i++) {
		if (in[i] == value)
			return i;
	}
	return -1;
}

float weighted_tsne::calculate_seq_error_max(int k, int N) {

	float res = 0;

	for (int i = 0; i < k; i++) {
		res += (k - i) * std::abs(i - (N - i));
	}

	return res;
}

void weighted_tsne::calculate_sequence_error(std::vector<int> &NN1, std::vector<int> &NN2, std::vector<float> &errors, int N, int d, int k) {
	assert(d == N + 1);
	assert(k <= d);
	errors.resize(N, 0);

	// Calculate seq error for every data point
	for (int i = 0; i < N; i++)
	{
		// Extract the neighbourhood list from the list of neighbours (and skip the first one)
		int start_index = i * d + 1;
		int end_index = start_index + k;

		// Store the neighbourhoods in a new array for convenience
		std::vector<int> NN1_i(NN1.begin() + start_index, NN1.begin() + end_index); // +1 to skip itself (which is closest neighbour)
		std::vector<int> NN2_i(NN2.begin() + start_index, NN2.begin() + end_index);

		errors[i] = 0;

		for (int j = 0; j < k; j++) {
			// Find the rank of point j in the other neighbourhood list
			int rank_ji_1 = indexOf(NN1_i, NN2_i[j]);
			int rank_ji_2 = indexOf(NN2_i, NN1_i[j]);

			errors[i] += 0.5 * (k - j) * abs(j - rank_ji_1) + 0.5 * (k - j) * abs(j - rank_ji_2);
		}
	}
}

// From: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C++
int weighted_tsne::levenshtein_distance(const std::vector<int> &s1, const std::vector<int> &s2)
{
	// To change the type this function manipulates and returns, change
	// the return type and the types of the two variables below.
	int s1len = s1.size();
	int s2len = s2.size();

	auto column_start = (decltype(s1len))1;

	auto column = new decltype(s1len)[s1len + 1];
	std::iota(column + column_start - 1, column + s1len + 1, column_start - 1);

	for (auto x = column_start; x <= s2len; x++) {
		column[0] = x;
		auto last_diagonal = x - column_start;
		for (auto y = column_start; y <= s1len; y++) {
			auto old_diagonal = column[y];
			auto possibilities = {
				column[y] + 1,
				column[y - 1] + 1,
				last_diagonal + (s1[y - 1] == s2[x - 1] ? 0 : 1)
			};
			column[y] = std::min(possibilities);
			last_diagonal = old_diagonal;
		}
	}
	auto result = column[s1len];
	delete[] column;
	return result;
}

void weighted_tsne::calculate_levenshtein_error(std::vector<int> &NN1, std::vector<int> &NN2, std::vector<scalar_type> &errors, int N, int d) {

	int k = d - 1;
	errors.resize(N, 0);

	// Calculate seq error for every data point
	for (int i = 0; i < N; i++)
	{
		// Extract the neighbourhood list from the list of neighbours (and skip the first one)
		int start_index = i * d + 1;
		int end_index = start_index + k;

		// Store the neighbourhoods in a new array for convenience
		std::vector<int> NN1_i(NN1.begin() + start_index, NN1.begin() + end_index); // +1 to skip itself (which is closest neighbour)
		std::vector<int> NN2_i(NN2.begin() + start_index, NN2.begin() + end_index);

		errors[i] = levenshtein_distance(NN1_i, NN2_i);
	}
}

void weighted_tsne::compute_neighbours(std::vector<float> data, int N, int d, int k, std::vector<int> &res) {

	std::vector<weighted_tsne::scalar_type> distances_squared;
	hdi::dr::HDJointProbabilityGenerator<weighted_tsne::scalar_type>::Parameters temp_prob_gen_param;
	std::vector<float> temp_perplexity(N, k);
	temp_prob_gen_param._perplexities = temp_perplexity;
	temp_prob_gen_param._perplexity_multiplier = 1;

	// computeHighDimensionalDistances includes the point itself as its nearest neighbour
	prob_gen.computeHighDimensionalDistances(data.data(), d, N, distances_squared, res, temp_prob_gen_param);
}

void weighted_tsne::compute_weight_falloff(std::vector<float> in_data, int N, int d, std::set<int> selected_indices, int k, std::vector<float> &weights_falloff) {
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