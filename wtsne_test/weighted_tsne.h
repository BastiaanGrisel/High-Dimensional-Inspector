#include "hdi/dimensionality_reduction/tsne.h"
#include "hdi/utils/cout_log.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/data/embedding.h"
#include "hdi/data/panel_data.h"
#include "hdi/data/io.h"
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/scoped_timers.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QIcon>

#include <omp.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <set>
#include <windows.h>

class weighted_tsne {
public:
	typedef float scalar_type;
	typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_scalar_matrix;

	std::vector<scalar_type> data;

	hdi::dr::SparseTSNEUserDefProbabilities<scalar_type> tSNE;
	hdi::dr::SparseTSNEUserDefProbabilities<scalar_type>::Parameters tSNE_param;

	hdi::dr::HDJointProbabilityGenerator<scalar_type> prob_gen;
	hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_param;

	hdi::data::Embedding<scalar_type> embedding;

	int initialise_tsne(std::wstring data_path, int num_data_points, int num_dimensions);

	void do_iteration();

	void calculate_set_error(std::vector<int> &NN1, std::vector<int> &NN2, std::vector<scalar_type> &errors, int N, int d);
	float jaccard_similarity(std::vector<int> A, std::vector<int> B);
	void compute_neighbours(std::vector<float> data, int N, int d, int k, std::vector<int> &res);
};