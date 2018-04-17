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
	wt->tSNE_param._embedding_dimensionality = output_dims;

	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-1k.bin", N, input_dims);

	// Do iterations
	for (int i = 0; i < 200; i++) {
		hdi::utils::secureLogValue(&log, "Iteration", i);
		wt->do_iteration();
	}

	// Set selected points
	

	// Do iterations
	for (int i = 0; i < 200; i++) {
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
	test_create_embedding();
	system("pause");
}