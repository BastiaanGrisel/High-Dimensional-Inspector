#include "weighted_tsne.h"

void save_as_csv(std::vector<float> data, int N, int output_dims, std::string filename) {

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

void save_as_csv(std::vector<int> data, int N, int output_dims, std::string filename) {

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

void test_create_embedding() {
	hdi::utils::CoutLog log;

	weighted_tsne* wt = new weighted_tsne();

	// Set tSNE parameters
	int N = 1000;
	int input_dims = 784;
	int output_dims = 2;
	int iterations = 2000;

	wt->prob_gen_param._perplexity = 40;

	//wt->tSNE.setTheta(0.5); // Barnes-hut
	wt->tSNE.setTheta(0); // Exact
	//wt->tSNE.setTheta(0.0001); // Almost exact but BH

	wt->tSNE_param._mom_switching_iter = 250;
	wt->tSNE_param._remove_exaggeration_iter = 250;
	wt->tSNE_param._embedding_dimensionality = output_dims;

	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-1k.bin", N, input_dims);

	// No selected points
	//std::vector<int> selectedPoints{};

	// MNIST 10k
	//std::set<int> selectedPoints{ 5, 22, 23, 26, 34, 36, 37, 39, 43, 55, 73, 92, 98, 102, 115, 117, 129, 135, 171, 202, 211, 240, 243, 257, 259, 267, 271, 273, 275, 277, 279, 282, 284, 296, 306, 309, 335, 339, 342, 349, 352, 354, 365, 366, 369, 390, 398, 405, 406, 412, 432, 444, 461, 470, 495, 496, 498, 514, 521, 538, 539, 564, 606, 614, 630, 634, 635, 666, 667, 677, 678, 696, 701, 704, 711, 712, 727, 740, 745, 746, 771, 795, 796, 801, 804, 807, 819, 876, 879, 880, 909, 914, 925, 930, 934, 999, 1020, 1030, 1067, 1081, 1082, 1091, 1095, 1103, 1126, 1141, 1145, 1147, 1148, 1163, 1188, 1196, 1202, 1208, 1214, 1289, 1302, 1321, 1330, 1337, 1342, 1348, 1367, 1375, 1389, 1390, 1402, 1411, 1414, 1427, 1434, 1437, 1461, 1462, 1468, 1477, 1501, 1512, 1533, 1546, 1571, 1573, 1578, 1580, 1587, 1608, 1609, 1610, 1611, 1612, 1614, 1620, 1639, 1676, 1693, 1694, 1703, 1713, 1720, 1723, 1735, 1743, 1779, 1783, 1791, 1795, 1810, 1814, 1828, 1834, 1837, 1845, 1846, 1850, 1862, 1865, 1885, 1917, 1931, 1935, 1941, 1944, 1966, 1967, 1972, 1976, 1980, 1984, 1996, 2003, 2007, 2009, 2011, 2028, 2029, 2030, 2042, 2051, 2058, 2066, 2070, 2078, 2082, 2100, 2120, 2142, 2153, 2167, 2178, 2184, 2199, 2201, 2211, 2215, 2225, 2239, 2244, 2266, 2268, 2274, 2279, 2293, 2296, 2300, 2316, 2322, 2328, 2343, 2347, 2351, 2361, 2369, 2376, 2385, 2390, 2395, 2402, 2403, 2404, 2435, 2449, 2450, 2461, 2463, 2467, 2478, 2495, 2498, 2503, 2504, 2507, 2510, 2537, 2563, 2575, 2588, 2596, 2598, 2599, 2611, 2615, 2636, 2645, 2649, 2660, 2662, 2678, 2687, 2688, 2698, 2703, 2723, 2727, 2730, 2737, 2744, 2752, 2761, 2764, 2766, 2767, 2769, 2774, 2777, 2782, 2809, 2831, 2837, 2859, 2864, 2866, 2870, 2878, 2884, 2887, 2889, 2895, 2898, 2903, 2904, 2920, 2929, 2943, 2948, 2952, 2953, 2955, 2956, 2976, 2979, 2984, 2986, 2995, 2996, 3002, 3009, 3016, 3034, 3040, 3046, 3057, 3067, 3072, 3077, 3078, 3079, 3090, 3106, 3112, 3114, 3115, 3124, 3131, 3144, 3165, 3166, 3178, 3186, 3192, 3198, 3213, 3222, 3236, 3242, 3243, 3248, 3262, 3276, 3281, 3294, 3312, 3326, 3349, 3353, 3360, 3365, 3366, 3379, 3380, 3387, 3391, 3407, 3462, 3466, 3484, 3499, 3521, 3534, 3536, 3542, 3576, 3578, 3583, 3589, 3603, 3609, 3618, 3621, 3623, 3638, 3644, 3653, 3654, 3684, 3686, 3706, 3722, 3733, 3751, 3755, 3766, 3779, 3793, 3804, 3821, 3852, 3882, 3906, 3925, 3935, 3959, 3970, 3973, 3976, 4011, 4012, 4035, 4050, 4052, 4053, 4059, 4062, 4076, 4087, 4089, 4090, 4101, 4115, 4133, 4144, 4163, 4176, 4198, 4201, 4204, 4207, 4220, 4223, 4241, 4247, 4254, 4258, 4260, 4273, 4274, 4289, 4294, 4305, 4316, 4328, 4338, 4357, 4362, 4367, 4375, 4376, 4380, 4388, 4395, 4396, 4407, 4410, 4423, 4426, 4428, 4436, 4468, 4487, 4490, 4498, 4499, 4519, 4521, 4525, 4530, 4531, 4533, 4537, 4551, 4569, 4577, 4579, 4591, 4601, 4618, 4627, 4639, 4651, 4667, 4672, 4679, 4684, 4719, 4728, 4737, 4739, 4767, 4775, 4779, 4787, 4807, 4822, 4823, 4834, 4835, 4839, 4840, 4842, 4843, 4851, 4856, 4866, 4887, 4908, 4921, 4922, 4950, 4959, 4961, 4994, 5009, 5019, 5022, 5040, 5042, 5048, 5082, 5095, 5102, 5107, 5108, 5121, 5127, 5129, 5134, 5148, 5152, 5168, 5190, 5199, 5202, 5204, 5208, 5227, 5233, 5254, 5283, 5285, 5317, 5343, 5347, 5348, 5356, 5358, 5372, 5375, 5379, 5382, 5383, 5389, 5405, 5423, 5427, 5433, 5436, 5460, 5464, 5466, 5477, 5486, 5494, 5497, 5498, 5499, 5502, 5514, 5517, 5530, 5535, 5540, 5553, 5566, 5579, 5580, 5593, 5629, 5641, 5648, 5657, 5658, 5660, 5672, 5684, 5699, 5708, 5716, 5739, 5759, 5779, 5786, 5800, 5812, 5820, 5828, 5833, 5834, 5837, 5848, 5850, 5854, 5856, 5876, 5887, 5890, 5893, 5894, 5896, 5898, 5899, 5900, 5924, 5926, 5944, 5948, 5950, 5951, 5960, 5965, 5968, 5973, 5974, 5979, 6005, 6016, 6022, 6053, 6062, 6064, 6070, 6073, 6074, 6079, 6109, 6120, 6132, 6136, 6183, 6186, 6190, 6202, 6206, 6208, 6213, 6217, 6235, 6241, 6272, 6283, 6287, 6296, 6304, 6313, 6314, 6329, 6333, 6335, 6354, 6362, 6377, 6408, 6430, 6438, 6448, 6456, 6457, 6475, 6484, 6501, 6526, 6536, 6545, 6563, 6599, 6602, 6603, 6606, 6608, 6623, 6627, 6641, 6646, 6670, 6674, 6697, 6719, 6720, 6724, 6729, 6755, 6764, 6774, 6786, 6787, 6822, 6829, 6838, 6840, 6857, 6869, 6878, 6881, 6887, 6888, 6890, 6910, 6912, 6916, 6928, 6955, 6959, 6974, 6985, 6988, 6994, 6995, 7002, 7007, 7029, 7041, 7048, 7052, 7054, 7068, 7092, 7098, 7110, 7113, 7122, 7130, 7150, 7155, 7169, 7190, 7196, 7201, 7209, 7222, 7242, 7251, 7253, 7259, 7270, 7271, 7288, 7292, 7309, 7325, 7326, 7368, 7374, 7390, 7391, 7394, 7431, 7457, 7475, 7483, 7486, 7490, 7503, 7506, 7547, 7551, 7566, 7570, 7571, 7575, 7578, 7581, 7582, 7596, 7611, 7621, 7637, 7651, 7673, 7690, 7698, 7720, 7739, 7755, 7777, 7793, 7809, 7814, 7828, 7849, 7853, 7854, 7856, 7888, 7893, 7910, 7919, 7928, 7939, 7945, 7963, 7979, 8011, 8038, 8054, 8058, 8082, 8089, 8114, 8118, 8121, 8126, 8128, 8138, 8155, 8156, 8168, 8169, 8187, 8193, 8196, 8244, 8245, 8257, 8261, 8266, 8269, 8279, 8281, 8284, 8286, 8289, 8317, 8319, 8321, 8341, 8358, 8376, 8411, 8412, 8414, 8419, 8426, 8428, 8429, 8433, 8441, 8452, 8484, 8485, 8493, 8525, 8545, 8548, 8553, 8559, 8571, 8584, 8603, 8608, 8611, 8628, 8644, 8647, 8660, 8666, 8670, 8674, 8686, 8692, 8694, 8709, 8716, 8726, 8731, 8736, 8739, 8742, 8752, 8779, 8783, 8801, 8804, 8811, 8812, 8820, 8845, 8877, 8884, 8907, 8918, 8966, 8972, 8994, 9033, 9037, 9042, 9079, 9082, 9090, 9106, 9108, 9131, 9166, 9172, 9180, 9211, 9220, 9224, 9227, 9255, 9263, 9278, 9294, 9313, 9317, 9333, 9336, 9354, 9358, 9374, 9402, 9421, 9425, 9434, 9436, 9442, 9448, 9452, 9458, 9460, 9469, 9476, 9480, 9485, 9499, 9500, 9503, 9519, 9528, 9534, 9545, 9555, 9567, 9594, 9601, 9610, 9616, 9626, 9638, 9656, 9675, 9710, 9717, 9730, 9735, 9736, 9745, 9748, 9771, 9775, 9783, 9786, 9789, 9802, 9810, 9811, 9820, 9822, 9838, 9845, 9847, 9856, 9859, 9862, 9868, 9877, 9892, 9894, 9908, 9911, 9919, 9920, 9926, 9932, 9947, 9956, 9962, 9963, 9968 };
	
	// MNIST 1k
	//std::vector<int> selectedPoints{ 4,7,8,23,34,35,58,60,65,84,93,106,126,129,148,152,160,163,183,193,210,213,230,232,234,240,244,262,269,272,287,289,290,299,300,335,338,346,348,360,361,369,375,376,381,385,393,398,401,402,404,407,408,421,426,447,448,468,483,491,499,525,526,533,544,559,561,572,594,598,602,611,635,636,639,644,686,697,716,735,749,750,761,762,766,771,772,776,804,808,827,829,845,867,871,872,882,917,922,941,970,975,981,985,996 };
	std::vector<int> selectedPoints{ 4,7,8,23,34,35,58,60,65,84,93,106,126,129,148,152,160,163,183,193,210,213,230,232,234,240,244,262,269,272,287,289,290,299,300,335,338,346,348,360 };
	//std::vector<int> selectedPoints{ 4 };
	
	save_as_csv(selectedPoints, selectedPoints.size(), 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/selection.csv");

	std::set<int> selectedPointNeighbours;

	// Use actual n_neighbours
	
	int n_neighbours = 50;
	std::vector<weighted_tsne::scalar_type> distances_squared;
	std::vector<int> neighbour_indices;

	hdi::dr::HDJointProbabilityGenerator<weighted_tsne::scalar_type>::Parameters temp_prob_gen_param;
	temp_prob_gen_param._perplexity = n_neighbours; // computeHighDimensionalDistances actually calculates (n_neighbours + 1) but includes itself
	temp_prob_gen_param._perplexity_multiplier = 1;

	wt->prob_gen.computeHighDimensionalDistances(wt->data.data(), input_dims, N, distances_squared, neighbour_indices, temp_prob_gen_param);

	// Assemble data structure to hold weights
	std::vector<weighted_tsne::scalar_type> connection_weights;
	connection_weights.resize(N * N);

	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			connection_weights[i * N + j] = 1;
			connection_weights[j * N + i] = 1;
		}
	}*/

	//float selectedWeight = 0.5 / (selectedPoints.size() * n_neighbours);
	//float unselectedWeight = 0.5 / (N*(N-1) - selectedPoints.size() * n_neighbours);

	float selectedWeight = 1.0;
	float unselectedWeight = 0.5;
	float changethistoupdate = 122432323343434324;

	hdi::utils::secureLogValue(&log, "Selected weight", selectedWeight);
	hdi::utils::secureLogValue(&log, "Unselected weight", unselectedWeight);

	// Initialise the weights
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			connection_weights[i * N + j] = unselectedWeight;
			connection_weights[j * N + i] = unselectedWeight;
		}
	}

	//// Set weights from selected points to their nearest neighbours
	//for (int selected_point : selectedPoints) {
	//	for (int i = 0; i < n_neighbours; i++) {
	//		int neighbour_idx = neighbour_indices[selected_point * (n_neighbours + 1) + i + 1];
	//		connection_weights[selected_point * N + neighbour_idx] = selectedWeight; // i + 1 because it includes itself as the first NN
	//		connection_weights[neighbour_idx * N + selected_point] = selectedWeight;
	//	}
	//}

	// Use P to select the neighbourhood
	weighted_tsne::sparse_scalar_matrix P = wt->tSNE.getDistributionP();
	//std::vector<hdi::data::MapMemEff<uint32_t, float>>

	float neighbours_thres = 0.0001;

	for (int selected_point : selectedPoints) {
		for (auto elem : P[selected_point]) {
			if (elem.second > neighbours_thres) {
				connection_weights[selected_point * N + elem.first] = selectedWeight; // i + 1 because it includes itself as the first NN
				connection_weights[elem.first * N + selected_point] = selectedWeight;
			}
		}
	}

	wt->tSNE.setWeights(connection_weights);
	save_as_csv(connection_weights, N, N, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/weights.csv");


	//for (int i : neighbour_indices) {
	//	selectedPointNeighbours.insert(i);
	//}

	//float neighbours_thres = 0;

	//// Get P and also include the nearest n_neighbours of the points in the selection
	//weighted_tsne::sparse_scalar_matrix P = wt->tSNE.getDistributionP();
	////std::vector<hdi::data::MapMemEff<uint32_t, float>>

	//for (int selectedIndex : selectedPoints) {
	//	for (auto elem : P[selectedIndex]) {
	//		if (elem.second > neighbours_thres) {
	//			selectedPointNeighbours.insert(elem.first);
	//		}
	//	}
	//}
/*
	hdi::utils::secureLogValue(&log, "Selected points", selectedPoints.size());
	hdi::utils::secureLogValue(&log, "Additional neighbours included", selectedPointNeighbours.size());*/

	

	//// POINT WEIGHTS
	//std::vector<float> pointWeights(N);
	////pointWeights.resize(N);

	//int j = 34454345;

	//float selectedWeight = 1;
	//float unselectedWeight = 0;

	//// Set all to default weight
	//for (int i = 0; i < N; i++) {
	//	pointWeights[i] = unselectedWeight;
	//}

	//// Set selected weight
	//for (int selectedIndex : selectedPoints) {
	//	pointWeights[selectedIndex] = selectedWeight;
	//}

	//// Set weight of selected n_neighbours
	//for (int selectedIndex : selectedPointNeighbours) {
	//	pointWeights[selectedIndex] = selectedWeight;
	//}

	//wt->tSNE.setWeights(pointWeights);

	// Do iterations
	float iteration_time = 0;

	{
		hdi::utils::ScopedTimer<float, hdi::utils::Milliseconds> timer(iteration_time);

		for (int i = 0; i < iterations; i++) {
			if(i > 0 && i % 100 == 0)
				hdi::utils::secureLogValue(&log, "Iteration", i);

			wt->do_iteration();
		}

		//for (int i = 0; i < iterations/2; i++) {
		//	if(i > 0 && i % 100 == 0)
		//		hdi::utils::secureLogValue(&log, "Iteration", i);

		//	wt->do_iteration();
		//}

		//// Initialise the weights
		//for (int i = 0; i < N; i++) {
		//	for (int j = 0; j < N; j++) {
		//		connection_weights[i * N + j] = unselectedWeight;
		//		connection_weights[j * N + i] = unselectedWeight;
		//	}
		//}

		//// Set weights from selected points to their nearest neighbours
		//for (int selected_point : selectedPoints) {
		//	for (int i = 0; i < n_neighbours; i++) {
		//		connection_weights[selected_point * N + neighbour_indices[i + 1]] = selectedWeight; // i + 1 because it includes itself as the first NN
		//		connection_weights[neighbour_indices[i + 1] * N + selected_point] = selectedWeight;
		//	}
		//}

		//wt->tSNE.setWeights(connection_weights);

		//for (int i = 0; i < iterations/2; i++) {
		//	if (i > 0 && i % 100 == 0)
		//		hdi::utils::secureLogValue(&log, "Iteration", i);

		//	wt->do_iteration();
		//}
	}

	hdi::utils::secureLogValue(&log, "Total iteration time (s): ", iteration_time / 1000.0f);
	hdi::utils::secureLogValue(&log, "Average iteration time (ms): ", iteration_time / (float) iterations);

	// Save as CSV
	std::vector<weighted_tsne::scalar_type> res = wt->embedding.getContainer();

	//std::ofstream out_file2("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding-" + std::to_string(selectedWeight) + "-" + std::to_string(unselectedWeight) + ".csv");
	
	save_as_csv(wt->embedding.getContainer(), N, output_dims, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding.csv");
	//std::ofstream out_file2("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding.csv");

	//for (int i = 0; i < N; i++) {
	//	std::string line = "";

	//	for (int d = 0; d < output_dims; d++) {
	//		int index = i * output_dims + d;
	//		line += std::to_string(res[index]);

	//		if (d < (output_dims - 1)) {
	//			line += ",";
	//		}
	//	}

	//	out_file2 << line << std::endl;
	//}

	//out_file2.close();

	delete wt;
}



int main() {
	//omp_set_num_threads(3);
	test_create_embedding();
	system("pause");
}