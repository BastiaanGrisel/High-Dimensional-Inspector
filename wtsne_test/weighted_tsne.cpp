#include "weighted_tsne.h"

int weighted_tsne::initialise_tsne(std::wstring data_path, int num_data_points, int num_dimensions)
{
	try {
		//       QCoreApplication app(argc, argv);
		//       QCoreApplication::setApplicationName("tSNE from distance matrix");
		//	QCoreApplication::setApplicationVersion("0.1");

		//	QCommandLineParser parser;
		//	parser.setApplicationDescription("Command line version of the tSNE algorithm");
		//	parser.addHelpOption();
		//	parser.addVersionOption();
		//	parser.addPositionalArgument("data", QCoreApplication::translate("main", "High dimensional data."));
		//	parser.addPositionalArgument("output", QCoreApplication::translate("main", "Output file."));
		//	parser.addPositionalArgument("num_data_points", QCoreApplication::translate("main", "Num of data-points."));
		//       parser.addPositionalArgument("num_dimensions", QCoreApplication::translate("main", "Num of dimensions."));

		//////////////////////////////////////////////////
		/////////////////   Arguments    /////////////////
		//////////////////////////////////////////////////
		//	// Verbose
		//	QCommandLineOption verbose_option(QStringList() << "o" << "verbose",
		//									 QCoreApplication::translate("main", "Verbose"));
		//	parser.addOption(verbose_option);

		//	// Iterations
		//	QCommandLineOption iterations_option(QStringList() << "i" << "iterations",
		//			QCoreApplication::translate("main", "Run the gradient for <iterations>."),
		//			QCoreApplication::translate("main", "iterations"));
		//	parser.addOption(iterations_option);

		//	//Dimensions
		//	QCommandLineOption target_dimensions_option(QStringList() << "d" << "target_dimensions",
		//			QCoreApplication::translate("main", "Reduce the dimensionality to <target_dimensions>."),
		//			QCoreApplication::translate("main", "target_dimensions"));
		//	parser.addOption(target_dimensions_option);

		//	//Exaggeration iter
		//	QCommandLineOption exaggeration_iter_option(QStringList() << "x" << "exaggeration_iter",
		//			QCoreApplication::translate("main", "Remove the exaggeration factor after <exaggeration_iter> iterations."),
		//			QCoreApplication::translate("main", "exaggeration_iter"));
		//	parser.addOption(exaggeration_iter_option);

		//       //Perplexity
		//       QCommandLineOption perplexity_option(QStringList() << "p" << "perplexity",
		//               QCoreApplication::translate("main", "Use perplexity value of <perplexity>."),
		//               QCoreApplication::translate("main", "perplexity"));
		//       parser.addOption(perplexity_option);

		//       //Perplexity
		//       QCommandLineOption theta_option(QStringList() << "t" << "theta",
		//               QCoreApplication::translate("main", "Use theta value of <theta> in the BH computation [0 <= t <= 1]."),
		//               QCoreApplication::translate("main", "theta"));
		//       parser.addOption(theta_option);

		//       //Save similarities
		//       QCommandLineOption save_similarities_option(QStringList() << "s" << "similarities",
		//               QCoreApplication::translate("main", "Save the similarity matrix P in <similarities>."),
		//               QCoreApplication::translate("main", "similarities"));
		//       parser.addOption(save_similarities_option);

		//	// Process the actual command line arguments given by the user
		//	parser.process(app);

		//	const QStringList args = parser.positionalArguments();
		//	// source is args.at(0), destination is args.at(1)

		//////////////////////////////////////////////////
		//////////////////////////////////////////////////
		//////////////////////////////////////////////////

		//       if(args.size()!=4){
		//		std::cout << "Not enough arguments!" << std::endl;
		//		return -1;
		//	}

		//	int num_data_points         = atoi(args.at(2).toStdString().c_str());
		//       int num_dimensions          = atoi(args.at(3).toStdString().c_str());

		//	bool verbose                = false;
		//	int iterations              = 1000;
		//	int exaggeration_iter       = 250;
		//	int perplexity              = 30;
		//       double theta                = 0.5;
		//       int num_target_dimensions   = 2;


		//	verbose     = parser.isSet(verbose_option);
		//	if(parser.isSet(iterations_option)){
		//		iterations  = atoi(parser.value(iterations_option).toStdString().c_str());
		//	}
		//	if(parser.isSet(exaggeration_iter_option)){
		//		exaggeration_iter = atoi(parser.value(exaggeration_iter_option).toStdString().c_str());
		//	}
		//       if(parser.isSet(perplexity_option)){
		//           perplexity = atoi(parser.value(perplexity_option).toStdString().c_str());
		//       }
		//       if(parser.isSet(theta_option)){
		//           theta = atof(parser.value(theta_option).toStdString().c_str());
		//           hdi::checkAndThrowRuntime(theta >= 0 && theta <= 1, "Invalid theta value");
		//       }
		//       if(parser.isSet(target_dimensions_option)){
		//           num_target_dimensions = atoi(parser.value(target_dimensions_option).toStdString().c_str());
		//           hdi::checkAndThrowRuntime(num_target_dimensions >= 1, "Invalid number of target dimensions");
		//       }
		//	if(verbose){
		//		std::cout << "===============================================" << std::endl;
		//		std::cout << "Arguments" << std::endl;
		//		std::cout << "\tHigh-Dim:\t\t" << args.at(0).toStdString() << std::endl;
		//		std::cout << "\tOutput:\t\t\t" << args.at(1).toStdString() << std::endl;
		//           std::cout << "\t#target_dimensions:\t\t\t" << num_target_dimensions << std::endl;
		//		std::cout << "\t# data-points:\t\t" << num_data_points << std::endl;
		//		std::cout << "Parameters" << std::endl;
		//		std::cout << "\tIterations:\t\t" << iterations << std::endl;
		//		std::cout << "\tExaggeration iter:\t" << exaggeration_iter << std::endl;
		//		std::cout << "\tPerplexity:\t\t" << perplexity << std::endl;
		//           std::cout << "\tTheta:\t\t" << theta << std::endl;
		//		std::cout << "===============================================" << std::endl;
		//	}


		////////////////////////////////////////////////
		////////////////////////////////////////////////
		////////////////////////////////////////////////

		float data_loading_time = 0;
		float similarities_comp_time = 0;
		float gradient_desc_comp_time = 0;
		float data_saving_time = 0;

		////////////////////////////////////////////////
		////////////////////////////////////////////////
		////////////////////////////////////////////////

		//typedef float scalar_type;
		//Input
		data.resize(num_data_points * num_dimensions);

		{
			hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_loading_time);
			std::ifstream input_file(data_path, std::ios::in | std::ios::binary | std::ios::ate);
			if (int(input_file.tellg()) != int(sizeof(scalar_type) * num_dimensions * num_data_points)) {
				std::cout << "Input file size doesn't agree with input parameters!" << std::endl;
				return 1;
			}
			input_file.seekg(0, std::ios::beg);
			input_file.read(reinterpret_cast<char*>(data.data()), sizeof(scalar_type) * num_dimensions * num_data_points);
			input_file.close();
		}

		////////////////////////////////////////////////
		////////////////////////////////////////////////
		////////////////////////////////////////////////

		hdi::utils::CoutLog log;
		hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type distributions;
		//hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_param;
		//hdi::dr::SparseTSNEUserDefProbabilities<scalar_type>::Parameters tSNE_param;
		//hdi::data::Embedding<scalar_type> embedding;

		{
			hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);
			//prob_gen_param._perplexities = perplexity;
			prob_gen.computeProbabilityDistributions(data.data(), num_dimensions, num_data_points, distributions, prob_gen_param);
		}


		{
			//hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);
			//tSNE_param._embedding_dimensionality = num_target_dimensions;
			//tSNE_param._mom_switching_iter = exaggeration_iter;
			//tSNE_param._remove_exaggeration_iter = exaggeration_iter;
			tSNE.initialize(distributions, &embedding, tSNE_param);
			
			//tSNE.setTheta(theta);

			//hdi::utils::secureLog(&log,"Computing gradient descent...");
			//for(int iter = 0; iter < iterations; ++iter){
			//    tSNE.doAnIteration();
			//    hdi::utils::secureLogValue(&log,"Iter",iter,verbose);
			//}
			//hdi::utils::secureLog(&log,"... done!");
		}

		////////////////////////////////////////////////
		////////////////////////////////////////////////
		////////////////////////////////////////////////

		//{
		//    //Output
		//    hdi::utils::ScopedTimer<float,hdi::utils::Seconds> timer(data_saving_time);
		//    {
		//        std::ofstream output_file (args.at(1).toStdString(), std::ios::out|std::ios::binary);
		//        output_file.write(reinterpret_cast<const char*>(embedding.getContainer().data()),sizeof(scalar_type)*embedding.getContainer().size());
		//    }
		//    if(parser.isSet(save_similarities_option)){
		//        std::ofstream output_file (parser.value(save_similarities_option).toStdString().c_str(), std::ios::out|std::ios::binary);
		//        hdi::data::IO::saveSparseMatrix(distributions,output_file);
		//    }
		//}

		////////////////////////////////////////////////
		////////////////////////////////////////////////
		////////////////////////////////////////////////

		hdi::utils::secureLogValue(&log, "Data loading (sec)", data_loading_time);
		hdi::utils::secureLogValue(&log, "Similarities computation (sec)", similarities_comp_time);
		//hdi::utils::secureLogValue(&log, "Gradient descent (sec)", gradient_desc_comp_time);
		//hdi::utils::secureLogValue(&log, "Data saving (sec)", data_saving_time);
		return 0;
	}
	catch (std::logic_error& ex) { std::cout << "Logic error: " << ex.what() << std::endl; }
	catch (std::runtime_error& ex) { std::cout << "Runtime error: " << ex.what() << std::endl; }
	catch (...) { std::cout << "An unknown error occurred" << std::endl;; }
	return 1;
}

void weighted_tsne::do_iteration() {
	tSNE.doAnIteration();
	//hdi::utils::CoutLog log;
	//hdi::utils::secureLogValue(&log, "Iter", iter, verbose);
}

std::vector<int> intersection(std::vector<int> &v1, std::vector<int> &v2)
{
	std::vector<int> v3;

	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());

	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v3));

	return v3;
}


float weighted_tsne::jaccard_similarity(std::vector<int> A, std::vector<int> B) {

	float sizeA = A.size();
	float sizeB = B.size();
	float sizeIntersection = intersection(A, B).size();

	return sizeIntersection / (sizeA + sizeB - sizeIntersection);
}

void weighted_tsne::calculate_set_error(std::vector<int> &NN1, std::vector<int> &NN2, std::vector<scalar_type> &errors, int N, int d) {

	int k = d - 1;
	errors.resize(N, 0);

	// Calculate seq error for every data point
	for (int i = 0; i < N; i++)
	{
		// Extract the neighbourhood list from the list of neighbours (and skip the first one)
		int start_index = i * d + 1;
		int end_index = start_index + k;

		// Store the neighbourhoods in a new array for convenience
		std::vector<int> NN1_i(NN1.begin() + start_index, NN1.begin() + end_index); // +1 to skip itself (which is closest neighbour)
		std::vector<int> NN2_i(NN2.begin() + start_index, NN2.begin() + end_index);

		errors[i] = 1 - jaccard_similarity(NN1_i, NN2_i);
	}
}

// From: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C++
int weighted_tsne::levenshtein_distance(const std::vector<int> &s1, const std::vector<int> &s2)
{
	// To change the type this function manipulates and returns, change
	// the return type and the types of the two variables below.
	int s1len = s1.size();
	int s2len = s2.size();

	auto column_start = (decltype(s1len))1;

	auto column = new decltype(s1len)[s1len + 1];
	std::iota(column + column_start - 1, column + s1len + 1, column_start - 1);

	for (auto x = column_start; x <= s2len; x++) {
		column[0] = x;
		auto last_diagonal = x - column_start;
		for (auto y = column_start; y <= s1len; y++) {
			auto old_diagonal = column[y];
			auto possibilities = {
				column[y] + 1,
				column[y - 1] + 1,
				last_diagonal + (s1[y - 1] == s2[x - 1] ? 0 : 1)
			};
			column[y] = std::min(possibilities);
			last_diagonal = old_diagonal;
		}
	}
	auto result = column[s1len];
	delete[] column;
	return result;
}

void weighted_tsne::calculate_levenshtein_error(std::vector<int> &NN1, std::vector<int> &NN2, std::vector<scalar_type> &errors, int N, int d) {

	int k = d - 1;
	errors.resize(N, 0);

	// Calculate seq error for every data point
	for (int i = 0; i < N; i++)
	{
		// Extract the neighbourhood list from the list of neighbours (and skip the first one)
		int start_index = i * d + 1;
		int end_index = start_index + k;

		// Store the neighbourhoods in a new array for convenience
		std::vector<int> NN1_i(NN1.begin() + start_index, NN1.begin() + end_index); // +1 to skip itself (which is closest neighbour)
		std::vector<int> NN2_i(NN2.begin() + start_index, NN2.begin() + end_index);

		errors[i] = levenshtein_distance(NN1_i, NN2_i);
	}
}

void weighted_tsne::compute_neighbours(std::vector<float> data, int N, int d, int k, std::vector<int> &res) {

	std::vector<weighted_tsne::scalar_type> distances_squared;
	hdi::dr::HDJointProbabilityGenerator<weighted_tsne::scalar_type>::Parameters temp_prob_gen_param;
	std::vector<float> temp_perplexity(N, k);
	temp_prob_gen_param._perplexities = temp_perplexity;
	temp_prob_gen_param._perplexity_multiplier = 1;

	// computeHighDimensionalDistances includes the point itself as its nearest neighbour
	prob_gen.computeHighDimensionalDistances(data.data(), d, N, distances_squared, res, temp_prob_gen_param);
}

void weighted_tsne::compute_weight_falloff(std::vector<float> in_data, int N, int d, std::set<int> selected_indices, int k, std::vector<float> &weights_falloff) {
	std::vector<int> nn;
	compute_neighbours(in_data, N, d, k, nn);// TODO: inly calculate neighbours of selectedIndices, not of all points

	std::vector<int> min_nn(N, INT_MAX);

	// For each point, save the k-nn distance to the closest selected point. Save the distance to the closest selected point.
	for (int selectedIndex : selected_indices) {
		for (int j = 0; j <= k; j++) {
			int idx = selectedIndex * (k + 1); // Index of the first nearest neighbour of selectedIndex in nn (which is selectedIndex itself)
			int nn_idx = nn[idx + j]; // Index of the j-th nearest neighbour of selectedIndex in min_nn
			min_nn[nn_idx] = std::min(min_nn[nn_idx], j);
		}
	}

	// Calculate weights from min_nn
	weights_falloff.resize(N, 0);

	for (int i = 0; i < N; i++) {
		if (min_nn[i] != INT_MAX) {
			weights_falloff[i] = (k - min_nn[i]) / (float)k;
		}
	}
}