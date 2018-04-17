#include "weighted_tsne.h"

void test_run_tsne() {
	weighted_tsne* wt = new weighted_tsne();

	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k.bin", 10000, 784);

	for (int i = 0; i < 100; i++) {
		wt->do_iteration();
	}

	

	delete wt;
}

int main() {
	test_run_tsne();
	system("pause");
}