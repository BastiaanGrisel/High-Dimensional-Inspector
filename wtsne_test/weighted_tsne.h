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

#define NOMINMAX
#include <windows.h>

#define NOMINMAX

class weighted_tsne {
public:
	typedef float scalar_type;
	typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_scalar_matrix;

	std::vector<scalar_type> data;
	int num_data_points;
	int input_dimensions;

	hdi::dr::SparseTSNEUserDefProbabilities<scalar_type> tSNE;
	hdi::dr::SparseTSNEUserDefProbabilities<scalar_type>::Parameters tSNE_param;

	hdi::dr::HDJointProbabilityGenerator<scalar_type> prob_gen;
	hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_param;

	hdi::data::Embedding<scalar_type> embedding;

	int initialise_tsne(std::wstring data_path, int num_data_points, int num_dimensions);
	int initialise_tsne(std::vector<scalar_type> data, int num_data_points, int num_dimensions);

	void do_iteration();

	void set_locked_points(std::vector<int> indices);
	void set_coordinates(std::vector<int> indices, std::vector<scalar_type> coordinates);

	void compute_neighbours(std::vector<float> data, int N, int d, int k, std::vector<int> &res);
	void compute_weight_falloff(std::vector<float> in_data, int N, int d, std::vector<int> selected_indices, int k, std::vector<float> &weights_falloff);

	int read_bin(std::wstring file_path, int N, int d, std::vector<weighted_tsne::scalar_type> &out);
	
	int read_csv(std::wstring file_path, int N, int d, std::vector<weighted_tsne::scalar_type> &out);
	void write_csv(std::vector<float> data, int N, int output_dims, std::string filename);
	void write_csv(std::vector<double> data, int N, int output_dims, std::string filename);

	void lerp(std::vector<float> from, std::vector<float> to, std::vector<float> &res, float alpha);
};