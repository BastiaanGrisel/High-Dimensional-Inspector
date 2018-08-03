/*
*
* Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
* EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
* BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
*
*/

#include "hdi/utils/cout_log.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/hierarchical_sne.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include <qimage.h>
#include <QApplication>
#include "hdi/visualization/scatterplot_canvas_qobj.h"
#include "hdi/visualization/scatterplot_drawer_fixed_color.h"
#include <iostream>
#include <fstream>
#include "hdi/data/panel_data.h"
#include "hdi/data/pixel_data.h"
#include "hdi/data/image_data.h"
#include "hdi/visualization/image_view_qobj.h"
#include "hdi/visualization/scatterplot_drawer_user_defined_colors.h"
#include "hdi/visualization/scatterplot_drawer_labels.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/graph_algorithms.h"
#include "hdi/data/embedding.h"
#include "hdi/visualization/controller_embedding_selection_qobj.h"
#include <QDir>
#include "hdi/utils/math_utils.h"
#include "hdi/visualization/multiple_image_view_qobj.h"
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "weighted_tsne.h"

int main(int argc, char *argv[]) {

	SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);


	try {
		typedef float scalar_type;
		typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_scalar_matrix;

		QApplication app(argc, argv);
		QIcon icon;
		icon.addFile(":/brick32.png");
		icon.addFile(":/brick128.png");
		app.setWindowIcon(icon);

		hdi::utils::CoutLog log;

		///////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////
		/*    if(argc != 4){
		hdi::utils::secureLog(&log,"Not enough input parameters...");
		return 1;
		}*/
		std::vector<QColor> color_per_digit;
		color_per_digit.push_back(qRgb(16, 78, 139));
		color_per_digit.push_back(qRgb(139, 90, 43));
		color_per_digit.push_back(qRgb(138, 43, 226));
		color_per_digit.push_back(qRgb(0, 128, 0));
		color_per_digit.push_back(qRgb(255, 150, 0));
		color_per_digit.push_back(qRgb(204, 40, 40));
		color_per_digit.push_back(qRgb(131, 139, 131));
		color_per_digit.push_back(qRgb(0, 205, 0));
		color_per_digit.push_back(qRgb(20, 20, 20));
		color_per_digit.push_back(qRgb(0, 150, 255));



		weighted_tsne* wt = new weighted_tsne();

		// Set tSNE parameters
		int N = 1000;
		int input_dims = 784;
		int output_dims = 2;
		int iterations = 2000;

		//wt->tSNE.setTheta(0.5); // Barnes-hut
		wt->tSNE.setTheta(0); // Exact

		wt->tSNE_param._mom_switching_iter = 250;
		wt->tSNE_param._remove_exaggeration_iter = 250;
		wt->tSNE_param._embedding_dimensionality = output_dims;

		wt->prob_gen_param._perplexity = 40;

		// Load the entire dataset
		std::vector<weighted_tsne::scalar_type> data;
		wt->read_bin(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-subset-538.bin", N, input_dims, data);

		std::vector<weighted_tsne::scalar_type> labels;
		wt->read_bin(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-subset-538-labels.bin", N, 1, labels);

		float iteration_time = 0;

		wt->initialise_tsne(data, N, input_dims);

		// Settings
		int selectedIndex = 64;

		std::vector<int> selectedIndices = { selectedIndex };

		auto P = wt->tSNE.getDistributionP();

		for (auto elem : P[selectedIndex]) {
			if (elem.second > 0.004) {
				selectedIndices.push_back(elem.first);
			}
		}
		

		hdi::utils::secureLogValue(&log, "Number of selected indices", selectedIndices.size());

		std::vector<float> selectedIndicesFloat(selectedIndices.begin(), selectedIndices.end());
		wt->write_csv(selectedIndicesFloat, selectedIndices.size(), 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/selection.csv");


		std::vector<float> weights(N*N, 0);

		for (int i : selectedIndices) {
			for (int j = 0; j < N; j++) {
				weights[i * N + j] = 1;
				weights[j * N + i] = 1;
			}
		}

		double weight_sum = 0;

		for (int i = 0; i < weights.size(); i++) {
			weight_sum += weights[i];
		}

		//weight_sum = weight_sum / (N * ((double)N - 1.0) / 2);

		//std::cout << weight_sum;

		//system("pause");

		wt->tSNE.weights = weights;
		wt->tSNE.weights_normalisation = weight_sum;

		//wt->tSNE.setEmbeddingCoordinates(selectedIndices, selectionEmbeddingFinal);
		//wt->tSNE.setLockedPoints(selectedIndices);

		//wt->tSNE.setWeights(one_weights, one_weights, one_weights, one_weights);

		/*
		hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type probability;
		hdi::dr::HDJointProbabilityGenerator<scalar_type> prob_gen;
		prob_gen.setLogger(&log);
		prob_gen.computeJointProbabilityDistribution(panel_data.getData().data(),num_dimensions,num_pics,probability);

		prob_gen.statistics().log(&log);*/

		/* hdi::data::Embedding<scalar_type> embedding;
		hdi::dr::SparseTSNEUserDefProbabilities<scalar_type> tSNE;
		hdi::dr::SparseTSNEUserDefProbabilities<scalar_type>::Parameters tSNE_params;
		tSNE.setLogger(&log);
		tSNE_params._seed = 1;
		tSNE.setTheta(0.5);
		tSNE.initializeWithJointProbabilityDistribution(probability,&embedding,tSNE_params);*/

		hdi::viz::ScatterplotCanvas viewer;
		viewer.setBackgroundColors(qRgb(240, 240, 240), qRgb(200, 200, 200));
		viewer.setSelectionColor(qRgb(50, 50, 50));
		viewer.resize(1000, 1000);
		viewer.show();

		//hdi::viz::MultipleImageView image_view;
		//image_view.setPanelData(&panel_data);
		//image_view.show();
		//image_view.updateView();

		//hdi::viz::ControllerSelectionEmbedding selection_controller;
		//selection_controller.setActors(&panel_data,&(wt->embedding),&viewer);
		//selection_controller.setLogger(&log);
		//selection_controller.initialize();
		//selection_controller.addView(&image_view);

		std::vector<uint32_t> flags(N, 0);
		std::vector<float> embedding_colors_for_viz(N * 3, 0);

		for (int i = 0; i < N; i++) {
			// Color black
			//QColor color(qRgb(0, 0, 0));

			// Color by label
			QColor color = color_per_digit[(int)labels[i]];

			// Color by weight
			//QColor color(qRgb(255, 10, 10));
			//color.setHsv(255, 255, selected_high[i]*127.0);

			embedding_colors_for_viz[i * 3 + 0] = color.redF();
			embedding_colors_for_viz[i * 3 + 1] = color.greenF();
			embedding_colors_for_viz[i * 3 + 2] = color.blueF();
		}

		// Color only selected
		//for (int index : selectedIndices) {
		//	QColor color(qRgb(255, 255, 0));

		//	embedding_colors_for_viz[index * 3 + 0] = color.redF();
		//	embedding_colors_for_viz[index * 3 + 1] = color.greenF();
		//	embedding_colors_for_viz[index * 3 + 2] = color.blueF();
		//}

		//for (int index : selectedIndices) {
		//	QColor color = qRgb(255, 50, 50);
		//	embedding_colors_for_viz[index * 3 + 0] = color.redF();
		//	embedding_colors_for_viz[index * 3 + 1] = color.greenF();
		//	embedding_colors_for_viz[index * 3 + 2] = color.blueF();
		//}

		//for(int i = 0; i < N; ++i){
		//    int label = labels[i];
		//    auto color = color_per_digit[label];
		//    embedding_colors_for_viz[i*3+0] = color.redF();
		//    embedding_colors_for_viz[i*3+1] = color.greenF();
		//    embedding_colors_for_viz[i*3+2] = color.blueF();
		//}

		hdi::viz::ScatterplotDrawerUsedDefinedColors drawer;
		drawer.initialize(viewer.context());
		drawer.setData(wt->embedding.getContainer().data(), embedding_colors_for_viz.data(), flags.data(), N);
		drawer.setAlpha(0.8);
		drawer.setPointSize(5);
		viewer.addDrawer(&drawer);

		int iter = 0;

		{
			hdi::utils::ScopedTimer<float, hdi::utils::Milliseconds> timer(iteration_time);

			while (iter < iterations) {
				wt->do_iteration();

				// Selection growing
				float alpha = (iter - 250) / 300.0;
				alpha = alpha > 1.0 ? 1.0 : (alpha < 0.0 ? 0.0 : alpha);

				//wt->tSNE.setEmbeddingCoordinates(selectedIndices, selectionEmbeddingFinal);

				// Set point location at every iteration < 800
				//wt->lerp(selectionEmbeddingStart, selectionEmbeddingFinal, selectionEmbeddingCurrent, alpha);
				//wt->tSNE.setEmbeddingCoordinates(selectedIndices, selectionEmbeddingCurrent);

				// Lerp the weights
				//wt->lerp(one_weights, selected_high, lerp_weights, alpha);
				//wt->tSNE.weights = lerp_weights;

				if (iter == iterations - 1) {
					//int k = 200;
					//std::vector<int> highDimNeighbours;
					//std::vector<int> lowDimNeighbours;

					//// highDimNeighbours, lowDimNeighbours, includes the point itself as its nearest neighbour
					//wt->compute_neighbours(wt->data, N, input_dims, k, highDimNeighbours);
					//wt->compute_neighbours(wt->embedding.getContainer(), N, output_dims, k, lowDimNeighbours);

					std::vector<double> errors(N, 0);
					//wt->calculate_percentage_error(lowDimNeighbours, highDimNeighbours, per_errors, N, k + 1, k);
					wt->tSNE.computeKullbackLeiblerDivergences(errors);

					wt->write_csv(errors, N, 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/errors.csv");
				}

				wt->write_csv(wt->embedding.getContainer(), N, output_dims, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding.csv");

				{//limits
					std::vector<scalar_type> limits;
					wt->embedding.computeEmbeddingBBox(limits, 0.25);
					auto tr = QVector2D(limits[1], limits[3]);
					auto bl = QVector2D(limits[0], limits[2]);
					/*		float lim = 15;
					auto tr = QVector2D(lim, lim);
					auto bl = QVector2D(-lim, -lim);*/
					viewer.setTopRightCoordinates(tr);
					viewer.setBottomLeftCoordinates(bl);

					/*	if ((iter % 100) == 0) {
					hdi::utils::secureLogValue(&log, "tr1", limits[1]);
					hdi::utils::secureLogValue(&log, "tr2", limits[3]);
					}*/
				}


				if ((iter % 1) == 0) {
					viewer.updateGL();
					hdi::utils::secureLogValue(&log, "Iter", iter);
				}

				QApplication::processEvents();
				++iter;
			}
		}

		hdi::utils::secureLogValue(&log, "Average iteration time (ms): ", iteration_time / (float)iter);

		return app.exec();
	}
	catch (std::logic_error& ex) { std::cout << "Logic error: " << ex.what(); }
	catch (std::runtime_error& ex) { std::cout << "Runtime error: " << ex.what(); }
	catch (...) { std::cout << "An unknown error occurred"; }
}