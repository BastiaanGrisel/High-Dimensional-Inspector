#include "weighted_tsne.h"

void test_create_embedding() {
	hdi::utils::CoutLog log;

	weighted_tsne* wt = new weighted_tsne();

	// Set tSNE parameters
	int N = 10000;
	int input_dims = 784;
	int output_dims = 2;
	int iterations = 500;

	wt->prob_gen_param._perplexity = 40;

	wt->tSNE.setTheta(0.5); // Barnes-hut
	//wt->tSNE.setTheta(0); // Exact
	//wt->tSNE.setTheta(0.0001); // Almost exact but BH

	wt->tSNE_param._mom_switching_iter = 250;
	wt->tSNE_param._remove_exaggeration_iter = 250;
	wt->tSNE_param._embedding_dimensionality = output_dims;

	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k.bin", N, input_dims);

	// Set a weight for one selected point
	//std::vector<int> selectedPoints{ 4,7,8,23,34,35,58,60,65,84,93,106,126,129,148,152,160,163,183,193,210,213,230,232,234,240,244,262,269,272,287,289,290,299,300,335,338,346,348,360,361,369,375,376,381,385,393,398,401,402,404,407,408,421,426,447,448,468,483,491,499,525,526,533,544,559,561,572,594,598,602,611,635,636,639,644,686,697,716,735,749,750,761,762,766,771,772,776,804,808,827,829,845,867,871,872,882,917,922,941,970,975,981,985,996 };
	//std::vector<int> selectedPoints{ 4 };
	std::vector<int> selectedPoints{ };

	// POINT WEIGHTS
	std::vector<float> pointWeights(N);
	//pointWeights.resize(N);

	int j = 243334434434;

	float selectedWeight = 1;
	float unselectedWeight = 1;

	for (int i = 0; i < N; i++) {
		pointWeights[i] = unselectedWeight;
	}

	for (int i = 0; i < selectedPoints.size(); i++) {
		pointWeights[selectedPoints[i]] = selectedWeight;
	}

	wt->tSNE.setWeights(pointWeights);

	// Do iterations
	float iteration_time = 0;

	{
		hdi::utils::ScopedTimer<float, hdi::utils::Milliseconds> timer(iteration_time);

		for (int i = 0; i < iterations; i++) {
			if(i > 0 && i % 100 == 0)
				hdi::utils::secureLogValue(&log, "Iteration", i);

			wt->do_iteration();
		}
	}

	hdi::utils::secureLogValue(&log, "Average iteration time (ms): ", iteration_time / (float) iterations);

	// Save as CSV
	std::vector<scalar_type> res = wt->embedding.getContainer();

	//std::ofstream out_file2("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding-" + std::to_string(selectedWeight) + "-" + std::to_string(unselectedWeight) + ".csv");
	std::ofstream out_file2("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding.csv");

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