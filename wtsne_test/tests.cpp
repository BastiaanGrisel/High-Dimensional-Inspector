#include "weighted_tsne.h"

void test_create_embedding() {
	hdi::utils::CoutLog log;

	weighted_tsne* wt = new weighted_tsne();

	// Set tSNE parameters
	int N = 1000;
	int input_dims = 784;
	int output_dims = 2;

	wt->prob_gen_param._perplexity = 40;

	//wt->tSNE.setTheta(0.5); // Barnes-hut
	wt->tSNE.setTheta(0); // Exact

	wt->tSNE_param._mom_switching_iter = 250;
	wt->tSNE_param._remove_exaggeration_iter = 250;
	wt->tSNE_param._embedding_dimensionality = output_dims;

	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-1k.bin", N, input_dims);

	// Set a weight for one selected point
	//std::vector<int> selectedPoints{ 4,7,8,23,34,35,58,60,65,84,93,106,126,129,148,152,160,163,183,193,210,213,230,232,234,240,244,262,269,272,287,289,290,299,300,335,338,346,348,360,361,369,375,376,381,385,393,398,401,402,404,407,408,421,426,447,448,468,483,491,499,525,526,533,544,559,561,572,594,598,602,611,635,636,639,644,686,697,716,735,749,750,761,762,766,771,772,776,804,808,827,829,845,867,871,872,882,917,922,941,970,975,981,985,996 };
	std::vector<int> selectedPoints{ 4 };

	// POINT WEIGHTS
	std::vector<float> pointWeights(N);
	//pointWeights.resize(N);

	int j = 23434;

	float selectedWeight = 1;
	float unselectedWeight = 0;

	for (int i = 0; i < N; i++) {
		pointWeights[i] = unselectedWeight;
	}

	for (int i = 0; i < selectedPoints.size(); i++) {
		pointWeights[selectedPoints[i]] = selectedWeight;
	}

	// CONNECTION WEIGHTS
	std::vector<scalar_type> connectionWeights(N*N);
	//connectionWeights.resize(N * N);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			//connectionWeights[i * N + j] = std::max(pointWeights[i], pointWeights[j]);
			connectionWeights[i * N + j] = pointWeights[i] * pointWeights[j];
		}
	}

	hdi::utils::secureLogValue(&log, "weight", connectionWeights[selectedPoints[0]]);

	wt->tSNE.setWeights(connectionWeights);

	//// Set all connectionWeights to 1
	//for (int ij = 0; ij < N * N; ij++) {
	//	connectionWeights[ij] = 1;
	//}

	//wt->tSNE.setWeights(connectionWeights);

	// Do iterations
	for (int i = 0; i < 1000; i++) {
		if(i % 100 == 0)
			hdi::utils::secureLogValue(&log, "Iteration", i);

		wt->do_iteration();
	}

	//hdi::utils::secureLog(&log, "Done with first iterations");

	// Select points in square around center
	//std::vector<int> selectedIndices;
	//float width, height;
	//width = height = 0.7;

	//for (int i = 0; i < N; i++) {
	//	if (wt->embedding.getContainer()[i * output_dims] < width && wt->embedding.getContainer()[i * output_dims + 1] < height) {
	//		selectedIndices.push_back(i);
	//	}
	//}

	//// Save selected indices to CSV
	//std::ofstream out_file("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/selected.csv");

	//for (int i = 0; i < selectedIndices.size(); i++) {
	//	out_file << std::to_string(selectedIndices[i]) << std::endl;
	//}

	//out_file.close();

	// Set connectionWeights
	//hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type connectionWeights;
	//std::vector<scalar_type> connectionWeights;
	//connectionWeights.resize(N * N);

	//// Set a weight for one selected point
	//int selectedpoint = 10;

	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++) {
	//		connectionWeights[i * N + j] = (i == selectedpoint || j == selectedpoint) ? 1 : 0.01;
	//	}
	//}
	//
	//wt->tSNE.setWeights(connectionWeights);

	//hdi::utils::secureLog(&log, "Weights are set");

	//// Do iterations
	//for (int i = 0; i < 500; i++) {
	//	if (i % 100 == 0)
	//		hdi::utils::secureLogValue(&log, "Iteration", i);

	//	wt->do_iteration();
	//}

	// Save as CSV
	std::vector<scalar_type> res = wt->embedding.getContainer();

	std::ofstream out_file2("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embefghgfhdding-" + std::to_string(selectedWeight) + "-" + std::to_string(unselectedWeight) + ".csv");

	for (int i = 0; i < N; i++) {
		std::string line = "";

		for (int d = 0; d < output_dims; d++) {
			int index = i * output_dims + d;
			line += std::to_string(res[index]);

			if (d < (output_dims - 1)) {
				line += ",";
			}
		}

		out_file2 << line << std::endl;
	}

	out_file2.close();

	delete wt;
}

int main() {
	//omp_set_num_threads(3);
	test_create_embedding();
	system("pause");
}