#include "weighted_tsne.h"

void test_create_embedding() {
	hdi::utils::CoutLog log;

	weighted_tsne* wt = new weighted_tsne();

	// Set tSNE parameters
	int N = 1000;
	int input_dims = 784;
	int output_dims = 2;

	wt->prob_gen_param._perplexity = 40;

	wt->tSNE.setTheta(0.5);

	wt->tSNE_param._mom_switching_iter = 250;
	wt->tSNE_param._remove_exaggeration_iter = 250;
	wt->tSNE_param._embedding_dimensionality = output_dims;

	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-1k.bin", N, input_dims);

	// Do iterations
	for (int i = 0; i < 2000; i++) {
		if(i % 100 == 0)
			hdi::utils::secureLogValue(&log, "Iteration", i);

		wt->do_iteration();
	}

	hdi::utils::secureLog(&log, "Done with first iterations");

	// Select points in square around center
	std::vector<int> selectedIndices;
	float width, height;
	width = height = 0.7;

	for (int i = 0; i < N; i++) {
		if (wt->embedding.getContainer()[i * output_dims] < width && wt->embedding.getContainer()[i * output_dims + 1] < height) {
			selectedIndices.push_back(i);
		}
	}

	// Save selected indices to CSV
	std::ofstream out_file("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/selected.csv");

	for (int i = 0; i < selectedIndices.size(); i++) {
		out_file << std::to_string(selectedIndices[i]) << std::endl;
	}

	out_file.close();

	// Set weights
	std::vector<scalar_type> weights;
	weights.resize(N);

	for (int i = 0; i < N; i++) {
		weights[i] = 0.00001;
	}

	for (int index : selectedIndices) {
		weights[index] = 1;
	}

	//wt->tSNE_param._point_weights = weights;
	wt->tSNE.updateWeights(weights);

	hdi::utils::secureLog(&log, "Weights are set");

	// Do iterations
	for (int i = 0; i < 2000; i++) {
		if (i % 100 == 0)
			hdi::utils::secureLogValue(&log, "Iteration", i);

		wt->do_iteration();
	}

	// Save as CSV
	std::vector<scalar_type> res = wt->embedding.getContainer();

	std::ofstream out_file("C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding.csv");

	for (int i = 0; i < N; i++) {
		std::string line = "";

		for (int d = 0; d < output_dims; d++) {
			int index = i * output_dims + d;
			line += std::to_string(res[index]);

			if (d < (output_dims - 1)) {
				line += ",";
			}
		}

		out_file << line << std::endl;
	}

	out_file.close();

	delete wt;
}

int main() {
	//omp_set_num_threads(3);
	test_create_embedding();
	system("pause");
}