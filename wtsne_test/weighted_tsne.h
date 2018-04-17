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

typedef float scalar_type;

class weighted_tsne {
public:
	hdi::dr::SparseTSNEUserDefProbabilities<scalar_type> tSNE;
	hdi::dr::SparseTSNEUserDefProbabilities<scalar_type>::Parameters tSNE_param;
	hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_param;

	hdi::data::Embedding<scalar_type> embedding;

	int initialise_tsne(std::wstring data_path, int num_data_points, int num_dimensions);

	void do_iteration();
};