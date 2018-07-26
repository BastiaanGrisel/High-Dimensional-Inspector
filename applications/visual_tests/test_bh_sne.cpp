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

int main(int argc, char *argv[]){

	SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);


    try{
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
        color_per_digit.push_back(qRgb(16,78,139));
        color_per_digit.push_back(qRgb(139,90,43));
        color_per_digit.push_back(qRgb(138,43,226));
        color_per_digit.push_back(qRgb(0,128,0));
        color_per_digit.push_back(qRgb(255,150,0));
        color_per_digit.push_back(qRgb(204,40,40));
        color_per_digit.push_back(qRgb(131,139,131));
        color_per_digit.push_back(qRgb(0,205,0));
        color_per_digit.push_back(qRgb(20,20,20));
        color_per_digit.push_back(qRgb(0, 150, 255));

  //      const int num_pics(10000);
  //      const int num_dimensions(784);

  //      std::ifstream file_data(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k.bin", std::ios::in|std::ios::binary);
  //      std::ifstream file_labels(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k-labels.bin", std::ios::in|std::ios::binary);
  //    
		//if (!file_labels.is_open()){
  //          throw std::runtime_error("label file cannot be found");
  //      }
  //      if (!file_data.is_open()){
  //          throw std::runtime_error("data file cannot be found");
  //      }

  //      //{//removing headers
  //      //    int32_t appo;
  //      //    file_labels.read((char*)&appo,4);
  //      //    file_labels.read((char*)&appo,4);
  //      //    file_data.read((char*)&appo,4);
  //      //    file_data.read((char*)&appo,4);
  //      //    file_data.read((char*)&appo,4);
  //      //    file_data.read((char*)&appo,4);
  //      //}

  //      hdi::data::PanelData<scalar_type> panel_data;
  //      {//initializing panel data
  //          for(int j = 0; j < 28; ++j){
  //              for(int i = 0; i < 28; ++i){
  //                  panel_data.addDimension(std::make_shared<hdi::data::PixelData>(hdi::data::PixelData(j,i,28,28)));
  //              }
  //          }
  //          panel_data.initialize();
  //      }


  //      std::vector<QImage> images;
  //      std::vector<std::vector<scalar_type> > input_data;
  //      std::vector<unsigned int> labels;

  //      {//reading data
  //          images.reserve(num_pics);
  //          input_data.reserve(num_pics);
  //          labels.reserve(num_pics);

  //          for(int i = 0; i < num_pics; ++i){
  //              unsigned char label;
  //              file_labels.read((char*)&label,1);
  //              labels.push_back(label);

  //              //still some pics to read for this digit
  //              input_data.push_back(std::vector<scalar_type>(num_dimensions));
  //              images.push_back(QImage(28,28,QImage::Format::Format_ARGB32));
  //              const int idx = int(input_data.size()-1);
  //              for(int i = 0; i < num_dimensions; ++i){
  //                  unsigned char pixel;
  //                  file_data.read((char*)&pixel,1);
  //                  const scalar_type intensity(255.f - pixel);
  //                  input_data[idx][i] = intensity;
  //                  images[idx].setPixel(i%28,i/28,qRgb(intensity,intensity,intensity));
  //              }
  //          }

  //          {
  //              //moving a digit at the beginning digits of the vectors
  //              const int digit_to_be_moved = 1;
  //              int idx_to_be_swapped = 0;
  //              for(int i = 0; i < images.size(); ++i){
  //                  if(labels[i] == digit_to_be_moved){
  //                      std::swap(images[i],		images[idx_to_be_swapped]);
  //                      std::swap(input_data[i],	input_data[idx_to_be_swapped]);
  //                      std::swap(labels[i],		labels[idx_to_be_swapped]);
  //                      ++idx_to_be_swapped;
  //                  }
  //              }
  //          }
  //          const int digit_to_be_selected = 4;
  //          for(int i = 0; i < images.size(); ++i){
  //              panel_data.addDataPoint(std::make_shared<hdi::data::ImageData>(hdi::data::ImageData(images[i])), input_data[i]);
  //              if(labels[i] == digit_to_be_selected){
  //                  //panel_data.getFlagsDataPoints()[i] = hdi::data::PanelData<scalar_type>::Selected;
  //              }
  //          }
  //      }

        //hdi::utils::secureLog(&log,"Data loaded...");

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

		weighted_tsne* wt = new weighted_tsne();

		// Set tSNE parameters
		int N = 10000;
		int input_dims = 784;
		int output_dims = 2;
		int iterations = 1000;

		wt->tSNE.setTheta(0.5); // Barnes-hut
		//wt->tSNE.setTheta(0); // Exact

		wt->tSNE_param._mom_switching_iter = 250;
		wt->tSNE_param._remove_exaggeration_iter = 250;
		wt->tSNE_param._embedding_dimensionality = output_dims;

		std::vector<float> perplexities(N, 40);
		wt->prob_gen_param._perplexities = perplexities;

		// Load the entire dataset
		std::vector<weighted_tsne::scalar_type> data;
		wt->read_bin(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k.bin", N, input_dims, data);

		std::vector<weighted_tsne::scalar_type> labels;
		wt->read_bin(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k-labels.bin", N, 1, labels);

		float iteration_time = 0;

		wt->initialise_tsne(data, N, input_dims);

		// Load selected points
		std::vector<weighted_tsne::scalar_type> selectedIndicesFloat;

		//int N_selected = 994;
		//wt->read_csv(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/labels-9-994.csv", N_selected, 1, selectedIndicesFloat);

		//int N_selected = 839;
		//wt->read_csv(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/mnist-10k-selection-358-839.csv", N_selected, 1, selectedIndicesFloat);

		//std::vector<int> selectedIndices(selectedIndicesFloat.begin(), selectedIndicesFloat.end());

		//std::vector<weighted_tsne::scalar_type> selectionEmbeddingFinal;
		//wt->read_csv(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding-selection.csv", N_selected, output_dims, selectionEmbeddingFinal);
		//std::vector<weighted_tsne::scalar_type> selectionEmbeddingStart(selectionEmbeddingFinal.size());
		//std::vector<weighted_tsne::scalar_type> selectionEmbeddingCurrent(selectionEmbeddingFinal.size());

		//for (int i = 0; i < selectionEmbeddingStart.size(); i++) {
		//	selectionEmbeddingStart[i] = 0.0001f * selectionEmbeddingFinal[i];
		//	selectionEmbeddingFinal[i] = 0.2f * selectionEmbeddingFinal[i];

		//	//if (i % 1 == 0) {
		//	//	selectionEmbeddingStart[i] += 1.0;
		//	//}
		//}

		std::vector<int> selectedIndices = { 4 };
		//std::vector<weighted_tsne::scalar_type> selectionEmbeddingStart = { 0, 0 };
		//std::vector<weighted_tsne::scalar_type> selectionEmbeddingEnd = { 0, 0 };


		//wt->set_coordinates(selectedIndices, selectionEmbeddingFinal);
		////wt->set_coordinates(selectedIndices, std::vector<float>{ 0,0 });
		//wt->set_locked_points(selectedIndices);


		// Add nearest neighbours to selection
		//int k = 50;
		//std::vector<int> nearestNeighbours;
		//wt->compute_neighbours(data, N, input_dims, k, nearestNeighbours);

		//std::set<int> selectedIndicesWithNeighbours;
		//for (int index : selectedIndices) {
		//	for (int i = 0; i < k; i++) {
		//		selectedIndicesWithNeighbours.insert(nearestNeighbours[index * (k + 1) + i]);
		//	}
		//}


		// Set weights
		std::vector<float> one_weights(N, 1);
		std::vector<float> zero_weights(N, 0);
		std::vector<float> high_weights(N, 2);
		std::vector<float> selected_high(N, 0);
		std::vector<float> selected_higher(N, 0);
		//std::vector<float> selected_high_extended(N, 0);
		std::vector<float> lerp_weights(N, 1);

		for (int index : selectedIndices) {
			selected_high[index] = 1;
			selected_higher[index] = 1000;
		}

	/*	for (int index : selectedIndicesWithNeighbours) {
			selected_high_extended[index] = 2;
		}
*/
		// Weight falloff
		//std::vector<float> weights_falloff;
		//wt->compute_weight_falloff(data, N, input_dims, selectedIndices, 1000, weights_falloff);
		//for (int i = 0; i < weights_falloff.size(); i++) {
		//	weights_falloff[i] = weights_falloff[i];
		//}

		//sparse_scalar_matrix s;
		//s.resize(N);

		//for (int index : selectedIndices) {
		//	for (int i = 0; i < N; i++) {
		//		//s[index][i] = 1;
		//		s[i][index] = 1;
		//	}
		//}

		//wt->tSNE.connection_weights = s;

		wt->tSNE.setWeights(one_weights, one_weights, one_weights, one_weights);

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
        viewer.setBackgroundColors(qRgb(240,240,240),qRgb(200,200,200));
        viewer.setSelectionColor(qRgb(50,50,50));
        viewer.resize(1000,1000);
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
        std::vector<float> embedding_colors_for_viz(N*3,0);

		for (int i = 0; i < N; i++) {
			// Color black
			//QColor color(qRgb(0, 0, 0));

			// Color by label
			QColor color = color_per_digit[(int) labels[i]];

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

			while (iter<1000) {
				wt->do_iteration();

				// Selection growing
				float alpha = (iter - 250) / 300.0;
				alpha = alpha > 1.0 ? 1.0 : (alpha < 0.0 ? 0.0 : alpha);

				// Set point location at every iteration < 800
				//wt->lerp(selectionEmbeddingStart, selectionEmbeddingFinal, selectionEmbeddingCurrent, alpha);
				//wt->set_coordinates(selectedIndices, selectionEmbeddingCurrent);

				// Lerp the weights
				//wt->lerp(one_weights, selected_high, lerp_weights, alpha);
				//wt->tSNE.setWeights(lerp_weights, lerp_weights, one_weights, one_weights);
				if (iter == 1000) {
					//int k = 200;
					//std::vector<int> highDimNeighbours;
					//std::vector<int> lowDimNeighbours;

					//// highDimNeighbours, lowDimNeighbours, includes the point itself as its nearest neighbour
					//wt->compute_neighbours(wt->data, N, input_dims, k, highDimNeighbours);
					//wt->compute_neighbours(wt->embedding.getContainer(), N, output_dims, k, lowDimNeighbours);

					std::vector<float> errors(N, 0);
					//wt->calculate_percentage_error(lowDimNeighbours, highDimNeighbours, per_errors, N, k + 1, k);
					wt->calculate_kl_divergence(errors);

					wt->write_csv(errors, N, 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/errors.csv");
				}

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
    catch(std::logic_error& ex){ std::cout << "Logic error: " << ex.what();}
    catch(std::runtime_error& ex){ std::cout << "Runtime error: " << ex.what();}
    catch(...){ std::cout << "An unknown error occurred";}
}
