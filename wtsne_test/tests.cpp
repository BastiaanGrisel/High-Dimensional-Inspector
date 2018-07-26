#include "weighted_tsne.h"


int main() {
	hdi::utils::CoutLog log;

	weighted_tsne* wt = new weighted_tsne();
	
	// Set tSNE parameters
	int N = 10000;
	int input_dims = 784;
	int output_dims = 2;
	int iterations = 1000;
	
	wt->tSNE.setTheta(0.5); // Barnes-hut
	//wt->tSNE.setTheta(0); // Exact
	//wt->tSNE.setTheta(0.0001); // Almost exact but BH
	
	wt->tSNE_param._mom_switching_iter = 250;
	wt->tSNE_param._remove_exaggeration_iter = 250;
	wt->tSNE_param._embedding_dimensionality = output_dims;

	wt->prob_gen_param._perplexity = 40;
	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k.bin", N, input_dims);
		
	float iteration_time = 0;
		
	// Run the actual algorithm
	{
		hdi::utils::ScopedTimer<float, hdi::utils::Milliseconds> timer(iteration_time);

		for (int i = 0; i < iterations; i++) {
			if (i > 0 && i % 100 == 0)
				hdi::utils::secureLogValue(&log, "Iteration", i);

			wt->do_iteration();
		}
	}
		
	hdi::utils::secureLogValue(&log, "Average iteration time", iteration_time/(float)iterations);

	system("pause");
}