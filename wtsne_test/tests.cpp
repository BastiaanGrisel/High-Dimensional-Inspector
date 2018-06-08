#include "weighted_tsne.h"

void save_as_csv(std::vector<float> data, int N, int output_dims, std::string filename) {

	std::ofstream out_file2(filename);

	for (int i = 0; i < N; i++) {
		std::string line = "";

		for (int d = 0; d < output_dims; d++) {
			int index = i * output_dims + d;
			line += std::to_string(data[index]);

			if (d < (output_dims - 1)) {
				line += ",";
			}
		}

		out_file2 << line << std::endl;
	}

	out_file2.close();
}

void save_as_csv(std::vector<int> data, int N, int output_dims, std::string filename) {

	std::ofstream out_file2(filename);
	out_file2 << "";

	for (int i = 0; i < N; i++) {
		std::string line = "";

		for (int d = 0; d < output_dims; d++) {
			int index = i * output_dims + d;
			line += std::to_string(data[index]);

			if (d < (output_dims - 1)) {
				line += ",";
			}
		}

		out_file2 << line << std::endl;
	}

	out_file2.close();
}


void test_jaccard_similarity() {
	std::vector<int> A = { 1, 2, 3, 4, 5 };
	std::vector<int> B = { 1, 2, 3, 6, 7 };

	weighted_tsne* wt = new weighted_tsne();

	std::cout << std::to_string(wt->jaccard_similarity(A, B));

	delete wt;
}

void test_create_embedding() {
	hdi::utils::CoutLog log;

	weighted_tsne* wt = new weighted_tsne();

	// Set tSNE parameters
	int N = 10000;
	int input_dims = 784;
	int output_dims = 2;
	int iterations = 1000;

	wt->prob_gen_param._perplexity = 40;

	wt->tSNE.setTheta(0.5); // Barnes-hut
	//wt->tSNE.setTheta(0); // Exact
	//wt->tSNE.setTheta(0.0001); // Almost exact but BH

	wt->tSNE_param._mom_switching_iter = 250;
	wt->tSNE_param._remove_exaggeration_iter = 250;
	wt->tSNE_param._embedding_dimensionality = output_dims;

	wt->initialise_tsne(L"C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/CSV-to-BIN/datasets-bin/mnist-10k.bin", N, input_dims);


	// STEP 1: Set selected points

	// No selected points
	//std::vector<int> selectedPoints{};

	// MNIST 10k
	// The zeros
	//std::set<int> selectedPoints{ 5, 22, 23, 26, 34, 36, 37, 39, 43, 55, 73, 92, 98, 102, 115, 117, 129, 135, 171, 202, 211, 240, 243, 257, 259, 267, 271, 273, 275, 277, 279, 282, 284, 296, 306, 309, 335, 339, 342, 349, 352, 354, 365, 366, 369, 390, 398, 405, 406, 412, 432, 444, 461, 470, 495, 496, 498, 514, 521, 538, 539, 564, 606, 614, 630, 634, 635, 666, 667, 677, 678, 696, 701, 704, 711, 712, 727, 740, 745, 746, 771, 795, 796, 801, 804, 807, 819, 876, 879, 880, 909, 914, 925, 930, 934, 999, 1020, 1030, 1067, 1081, 1082, 1091, 1095, 1103, 1126, 1141, 1145, 1147, 1148, 1163, 1188, 1196, 1202, 1208, 1214, 1289, 1302, 1321, 1330, 1337, 1342, 1348, 1367, 1375, 1389, 1390, 1402, 1411, 1414, 1427, 1434, 1437, 1461, 1462, 1468, 1477, 1501, 1512, 1533, 1546, 1571, 1573, 1578, 1580, 1587, 1608, 1609, 1610, 1611, 1612, 1614, 1620, 1639, 1676, 1693, 1694, 1703, 1713, 1720, 1723, 1735, 1743, 1779, 1783, 1791, 1795, 1810, 1814, 1828, 1834, 1837, 1845, 1846, 1850, 1862, 1865, 1885, 1917, 1931, 1935, 1941, 1944, 1966, 1967, 1972, 1976, 1980, 1984, 1996, 2003, 2007, 2009, 2011, 2028, 2029, 2030, 2042, 2051, 2058, 2066, 2070, 2078, 2082, 2100, 2120, 2142, 2153, 2167, 2178, 2184, 2199, 2201, 2211, 2215, 2225, 2239, 2244, 2266, 2268, 2274, 2279, 2293, 2296, 2300, 2316, 2322, 2328, 2343, 2347, 2351, 2361, 2369, 2376, 2385, 2390, 2395, 2402, 2403, 2404, 2435, 2449, 2450, 2461, 2463, 2467, 2478, 2495, 2498, 2503, 2504, 2507, 2510, 2537, 2563, 2575, 2588, 2596, 2598, 2599, 2611, 2615, 2636, 2645, 2649, 2660, 2662, 2678, 2687, 2688, 2698, 2703, 2723, 2727, 2730, 2737, 2744, 2752, 2761, 2764, 2766, 2767, 2769, 2774, 2777, 2782, 2809, 2831, 2837, 2859, 2864, 2866, 2870, 2878, 2884, 2887, 2889, 2895, 2898, 2903, 2904, 2920, 2929, 2943, 2948, 2952, 2953, 2955, 2956, 2976, 2979, 2984, 2986, 2995, 2996, 3002, 3009, 3016, 3034, 3040, 3046, 3057, 3067, 3072, 3077, 3078, 3079, 3090, 3106, 3112, 3114, 3115, 3124, 3131, 3144, 3165, 3166, 3178, 3186, 3192, 3198, 3213, 3222, 3236, 3242, 3243, 3248, 3262, 3276, 3281, 3294, 3312, 3326, 3349, 3353, 3360, 3365, 3366, 3379, 3380, 3387, 3391, 3407, 3462, 3466, 3484, 3499, 3521, 3534, 3536, 3542, 3576, 3578, 3583, 3589, 3603, 3609, 3618, 3621, 3623, 3638, 3644, 3653, 3654, 3684, 3686, 3706, 3722, 3733, 3751, 3755, 3766, 3779, 3793, 3804, 3821, 3852, 3882, 3906, 3925, 3935, 3959, 3970, 3973, 3976, 4011, 4012, 4035, 4050, 4052, 4053, 4059, 4062, 4076, 4087, 4089, 4090, 4101, 4115, 4133, 4144, 4163, 4176, 4198, 4201, 4204, 4207, 4220, 4223, 4241, 4247, 4254, 4258, 4260, 4273, 4274, 4289, 4294, 4305, 4316, 4328, 4338, 4357, 4362, 4367, 4375, 4376, 4380, 4388, 4395, 4396, 4407, 4410, 4423, 4426, 4428, 4436, 4468, 4487, 4490, 4498, 4499, 4519, 4521, 4525, 4530, 4531, 4533, 4537, 4551, 4569, 4577, 4579, 4591, 4601, 4618, 4627, 4639, 4651, 4667, 4672, 4679, 4684, 4719, 4728, 4737, 4739, 4767, 4775, 4779, 4787, 4807, 4822, 4823, 4834, 4835, 4839, 4840, 4842, 4843, 4851, 4856, 4866, 4887, 4908, 4921, 4922, 4950, 4959, 4961, 4994, 5009, 5019, 5022, 5040, 5042, 5048, 5082, 5095, 5102, 5107, 5108, 5121, 5127, 5129, 5134, 5148, 5152, 5168, 5190, 5199, 5202, 5204, 5208, 5227, 5233, 5254, 5283, 5285, 5317, 5343, 5347, 5348, 5356, 5358, 5372, 5375, 5379, 5382, 5383, 5389, 5405, 5423, 5427, 5433, 5436, 5460, 5464, 5466, 5477, 5486, 5494, 5497, 5498, 5499, 5502, 5514, 5517, 5530, 5535, 5540, 5553, 5566, 5579, 5580, 5593, 5629, 5641, 5648, 5657, 5658, 5660, 5672, 5684, 5699, 5708, 5716, 5739, 5759, 5779, 5786, 5800, 5812, 5820, 5828, 5833, 5834, 5837, 5848, 5850, 5854, 5856, 5876, 5887, 5890, 5893, 5894, 5896, 5898, 5899, 5900, 5924, 5926, 5944, 5948, 5950, 5951, 5960, 5965, 5968, 5973, 5974, 5979, 6005, 6016, 6022, 6053, 6062, 6064, 6070, 6073, 6074, 6079, 6109, 6120, 6132, 6136, 6183, 6186, 6190, 6202, 6206, 6208, 6213, 6217, 6235, 6241, 6272, 6283, 6287, 6296, 6304, 6313, 6314, 6329, 6333, 6335, 6354, 6362, 6377, 6408, 6430, 6438, 6448, 6456, 6457, 6475, 6484, 6501, 6526, 6536, 6545, 6563, 6599, 6602, 6603, 6606, 6608, 6623, 6627, 6641, 6646, 6670, 6674, 6697, 6719, 6720, 6724, 6729, 6755, 6764, 6774, 6786, 6787, 6822, 6829, 6838, 6840, 6857, 6869, 6878, 6881, 6887, 6888, 6890, 6910, 6912, 6916, 6928, 6955, 6959, 6974, 6985, 6988, 6994, 6995, 7002, 7007, 7029, 7041, 7048, 7052, 7054, 7068, 7092, 7098, 7110, 7113, 7122, 7130, 7150, 7155, 7169, 7190, 7196, 7201, 7209, 7222, 7242, 7251, 7253, 7259, 7270, 7271, 7288, 7292, 7309, 7325, 7326, 7368, 7374, 7390, 7391, 7394, 7431, 7457, 7475, 7483, 7486, 7490, 7503, 7506, 7547, 7551, 7566, 7570, 7571, 7575, 7578, 7581, 7582, 7596, 7611, 7621, 7637, 7651, 7673, 7690, 7698, 7720, 7739, 7755, 7777, 7793, 7809, 7814, 7828, 7849, 7853, 7854, 7856, 7888, 7893, 7910, 7919, 7928, 7939, 7945, 7963, 7979, 8011, 8038, 8054, 8058, 8082, 8089, 8114, 8118, 8121, 8126, 8128, 8138, 8155, 8156, 8168, 8169, 8187, 8193, 8196, 8244, 8245, 8257, 8261, 8266, 8269, 8279, 8281, 8284, 8286, 8289, 8317, 8319, 8321, 8341, 8358, 8376, 8411, 8412, 8414, 8419, 8426, 8428, 8429, 8433, 8441, 8452, 8484, 8485, 8493, 8525, 8545, 8548, 8553, 8559, 8571, 8584, 8603, 8608, 8611, 8628, 8644, 8647, 8660, 8666, 8670, 8674, 8686, 8692, 8694, 8709, 8716, 8726, 8731, 8736, 8739, 8742, 8752, 8779, 8783, 8801, 8804, 8811, 8812, 8820, 8845, 8877, 8884, 8907, 8918, 8966, 8972, 8994, 9033, 9037, 9042, 9079, 9082, 9090, 9106, 9108, 9131, 9166, 9172, 9180, 9211, 9220, 9224, 9227, 9255, 9263, 9278, 9294, 9313, 9317, 9333, 9336, 9354, 9358, 9374, 9402, 9421, 9425, 9434, 9436, 9442, 9448, 9452, 9458, 9460, 9469, 9476, 9480, 9485, 9499, 9500, 9503, 9519, 9528, 9534, 9545, 9555, 9567, 9594, 9601, 9610, 9616, 9626, 9638, 9656, 9675, 9710, 9717, 9730, 9735, 9736, 9745, 9748, 9771, 9775, 9783, 9786, 9789, 9802, 9810, 9811, 9820, 9822, 9838, 9845, 9847, 9856, 9859, 9862, 9868, 9877, 9892, 9894, 9908, 9911, 9919, 9920, 9926, 9932, 9947, 9956, 9962, 9963, 9968 };
	//std::set<int> selectedPoints{ 5, 22, 23, 26, 34, 36, 37, 39, 43, 55, 73, 92, 98, 102, 115, 117, 129, 135, 171, 202, 211, 240, 243, 257, 259, 267, 271, 273, 275, 277, 279, 282, 284, 296, 306, 309, 335, 339, 342, 349, 352, 354, 365, 366, 369, 390, 398 };
	
	// Some Threes
	std::set<int> selectedPoints{ 1,3,21,29,59,76,101,113,120,133,134,136,137,140,141,145,152,166,167,175,190,193,204,207,221,236,245,246,262,287,290,297,310,311,313,315,316,321,333,372,380,387,393,395,397,401,403,415,457,478,482,487,494,504,518,547,565,580,584,605,645,662,679,702,703,705,717,757,776,791,792,817,825,826,828,830,866,901,905,906,907,908,912,915,926,931,935,936,941,947,968,976,1003,1008,1029,1041,1058,1079,1090,1100,1115,1119,1124,1176,1177,1191,1197,1212,1230,1237,1239,1267,1278,1297,1305,1320,1323,1329,1338,1351,1356,1379,1380,1399,1407,1412,1418,1428,1429,1439,1441,1455,1465,1478,1481,1496,1508,1515,1526,1537,1540,1545,1554,1559,1563,1574,1586,1592,1605,1617,1621,1622,1627,1628,1633,1635,1638,1640,1644,1645,1656,1658,1666,1679,1684,1690,1698,1722,1732,1754,1766,1775,1787,1802,1809,1860,1879,1895,1908,1912,1918,1933,1936,1956,1977,2006,2022,2025,2034,2037,2050,2054,2073,2074,2077,2080,2110,2124,2133,2177,2188,2213,2216,2265,2269,2270,2281,2286,2288,2309,2310,2312,2318,2324,2339,2342,2346,2388,2391,2405,2407,2418,2419,2421,2429,2443,2471,2472,2474,2481,2490,2505,2506,2534,2545,2550,2562,2565,2583,2590,2672,2679,2692,2695,2719,2721,2734,2742,2753,2758,2781,2794,2797,2799,2805,2807,2811,2814,2852,2861,2873,2907,2914,2960,2968,2974,3007,3023,3045,3047,3064,3071,3102,3108,3117,3119,3121,3125,3137,3146,3158,3167,3174,3181,3188,3191,3206,3217,3220,3234,3250,3289,3313,3317,3321,3324,3327,3346,3369,3374,3378,3383,3388,3397,3400,3408,3420,3423,3457,3540,3557,3562,3572,3573,3586,3595,3601,3612,3630,3647,3651,3663,3671,3687,3702,3725,3727,3759,3763,3765,3770,3774,3778,3783,3805,3827,3832,3836,3855,3863,3864,3879,3889,3910,3917,3922,3928,3929,3939,3940,3953,3961,3969,3975,3983,3987,3988,3989,4004,4027,4029,4037,4057,4067,4082,4088,4126,4135,4136,4159,4177,4183,4187,4189,4190,4191,4200,4203,4206,4228,4235,4248,4271,4324,4337,4339,4360,4363,4374,4379,4381,4390,4391,4402,4412,4434,4444,4452,4475,4480,4503,4506,4512,4517,4539,4540,4560,4565,4568,4594,4600,4605,4615,4622,4625,4626,4628,4641,4650,4664,4674,4687,4702,4748,4752,4762,4781,4809,4812,4814,4858,4860,4867,4869,4890,4896,4898,4902,4916,4918,4931,4965,4966,4969,5013,5029,5030,5034,5035,5044,5061,5068,5075,5089,5096,5111,5137,5163,5179,5191,5215,5216,5223,5226,5241,5247,5264,5281,5284,5301,5321,5322,5324,5329,5355,5371,5391,5393,5396,5407,5413,5419,5429,5439,5445,5448,5468,5484,5493,5516,5546,5547,5555,5569,5573,5576,5581,5589,5605,5617,5661,5667,5669,5673,5687,5688,5690,5700,5706,5718,5735,5756,5760,5765,5768,5801,5808,5832,5844,5851,5867,5873,5883,5889,5911,5912,5927,5953,5964,5966,5972,6014,6041,6044,6061,6084,6088,6089,6119,6130,6134,6137,6144,6164,6209,6225,6228,6258,6263,6273,6274,6284,6299,6310,6323,6325,6328,6349,6351,6370,6381,6392,6394,6398,6400,6401,6406,6409,6413,6421,6449,6455,6469,6471,6491,6500,6502,6513,6516,6521,6525,6555,6566,6572,6573,6590,6593,6612,6614,6619,6621,6625,6631,6637,6643,6657,6663,6672,6692,6699,6716,6733,6762,6798,6807,6830,6853,6854,6855,6860,6873,6885,6896,6918,6927,6945,6960,6970,6977,6987,7000,7013,7019,7022,7040,7061,7063,7071,7076,7078,7083,7093,7095,7102,7105,7111,7118,7138,7144,7153,7158,7180,7183,7186,7218,7228,7231,7240,7287,7294,7299,7308,7310,7322,7340,7350,7352,7357,7369,7372,7373,7388,7392,7400,7413,7428,7442,7453,7458,7464,7471,7477,7488,7504,7508,7513,7519,7520,7525,7527,7531,7546,7595,7598,7603,7607,7631,7650,7652,7663,7680,7682,7719,7724,7725,7726,7737,7747,7762,7765,7769,7781,7797,7806,7816,7821,7825,7826,7842,7846,7864,7876,7885,7903,7904,7922,7943,7976,7980,7998,8013,8019,8020,8031,8067,8083,8096,8103,8108,8134,8135,8143,8145,8150,8152,8165,8182,8194,8198,8213,8239,8247,8252,8331,8332,8335,8337,8355,8357,8369,8377,8379,8398,8399,8405,8406,8416,8421,8438,8462,8470,8472,8473,8476,8487,8488,8502,8507,8510,8512,8513,8517,8522,8529,8533,8534,8540,8542,8543,8550,8554,8581,8589,8594,8595,8596,8606,8607,8624,8629,8640,8658,8676,8690,8697,8698,8705,8751,8755,8761,8787,8818,8829,8831,8835,8837,8839,8841,8844,8879,8900,8910,8914,8931,8938,8943,8946,8967,8998,8999,9004,9006,9016,9024,9047,9058,9059,9066,9101,9110,9116,9164,9167,9168,9209,9221,9230,9231,9236,9271,9280,9287,9312,9319,9322,9329,9334,9339,9345,9352,9377,9380,9389,9392,9401,9428,9443,9456,9470,9471,9475,9489,9492,9513,9520,9550,9560,9563,9566,9573,9577,9586,9592,9595,9600,9617,9623,9641,9642,9644,9650,9677,9689,9690,9703,9759,9791,9794,9807,9825,9846,9876,9883,9904,9907,9913,9929,9953,9955,9967,9971,9981,9985,9997 };

	// MNIST 1k
	//std::vector<int> selectedPoints{ 4,7,8,23,34,35,58,60,65,84,93,106,126,129,148,152,160,163,183,193,210,213,230,232,234,240,244,262,269,272,287,289,290,299,300,335,338,346,348,360,361,369,375,376,381,385,393,398,401,402,404,407,408,421,426,447,448,468,483,491,499,525,526,533,544,559,561,572,594,598,602,611,635,636,639,644,686,697,716,735,749,750,761,762,766,771,772,776,804,808,827,829,845,867,871,872,882,917,922,941,970,975,981,985,996 };
	//std::vector<int> selectedPoints{ 4 };

	// Save selected points
	hdi::utils::secureLogValue(&log, "Selected points", selectedPoints.size());
	std::vector<int> selectedIds(selectedPoints.begin(), selectedPoints.end());
	save_as_csv(selectedIds, selectedPoints.size(), 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/selection.csv");


	// STEP 2: Augment selected points

	std::set<int> selectedPointNeighbours;

	// 2.1 Using nearest neighbours
	//int neighbours = 5;
	//std::vector<weighted_tsne::scalar_type> distances_squared;
	//std::vector<int> indices;

	//hdi::dr::HDJointProbabilityGenerator<weighted_tsne::scalar_type>::Parameters temp_prob_gen_param;
	//temp_prob_gen_param._perplexity = neighbours - 1; // (it does +1 in HDJointProbabilityGenerator for some reason)
	//temp_prob_gen_param._perplexity_multiplier = 1;

	//wt->prob_gen.computeHighDimensionalDistances(wt->data.data(), input_dims, N, distances_squared, indices, temp_prob_gen_param);

	//for (int i : indices) {
	//	selectedPointNeighbours.insert(i);
	//}

	// 2.2 Using the p-values
	float neighbours_thres = 0.002;

	weighted_tsne::sparse_scalar_matrix P = wt->tSNE.getDistributionP(); //std::vector<hdi::data::MapMemEff<uint32_t, float>>
	
	for (int selectedIndex : selectedPoints) {
		for (auto elem : P[selectedIndex]) {
			if (elem.second > neighbours_thres) {
				selectedPointNeighbours.insert(elem.first);
			}
		}
	}


	// 2.3 Merge neighbors with selected points
	for (int selectedNeighbour : selectedPointNeighbours) {
		selectedPoints.insert(selectedNeighbour);
	}

	hdi::utils::secureLogValue(&log, "With extra neighbours included", selectedPoints.size());
	std::vector<int> selectedNeighbourIds(selectedPointNeighbours.begin(), selectedPointNeighbours.end());
	save_as_csv(selectedNeighbourIds, selectedPointNeighbours.size(), 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/selection-neighbours.csv");


	// STEP 3: Set the weights
	float j = 2523343513;
	float selectedWeight = 1.0f;
	float unselectedWeight = 0.01f;

	hdi::utils::secureLogValue(&log, "Selected weight", selectedWeight);
	hdi::utils::secureLogValue(&log, "Unselected weight", unselectedWeight);

	// 2.1 Set weights using set values
	std::vector<float> pointWeights(N, 1);
	std::vector<float> gradientWeights(N, unselectedWeight);

	// Set selected point weight
	//for (int selectedIndex : selectedPoints) {
	//	pointWeights[selectedIndex] = selectedWeight;
	//}

	// Set selected gradient weight
	for (int selectedIndex : selectedPoints) {
		gradientWeights[selectedIndex] = selectedWeight;
	}

	// Make gradient weights sum up to a constant
	float sum = 0;
	for (int i = 0; i < gradientWeights.size(); i++) sum += gradientWeights[i];
	for (int i = 0; i < gradientWeights.size(); i++) gradientWeights[i] = N * gradientWeights[i] / sum;


	// 2.2 Set weights based on P
	//std::vector<float> pointWeights(N, 1);
	//std::vector<float> gradientWeights(N, 1);

	//float weight = 100;

	//for (int selectedIndex : selectedPoints) {
	//	pointWeights[selectedIndex] += weight; // Add the sum of the p_ij values

	//	for (auto elem : P[selectedIndex]) { // elem.first is index, elem.second is p-value
	//		pointWeights[elem.first] += weight * elem.second;
	//	}
	//}

	wt->tSNE.setWeights(pointWeights, gradientWeights);
	save_as_csv(pointWeights, N, 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/point-weights.csv");
	save_as_csv(gradientWeights, N, 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/gradient-weights.csv");

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

	// Save embedding as CSV
	std::vector<weighted_tsne::scalar_type> res = wt->embedding.getContainer();
	save_as_csv(res, N, output_dims, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/embedding.csv");

	hdi::utils::secureLogValue(&log, "Total iteration time (s): ", iteration_time / 1000.0f);
	hdi::utils::secureLogValue(&log, "Average iteration time (ms): ", iteration_time / (float)iterations);

	// Calculate set error for each data point
	{
		int k = 50;
		std::vector<int> highDimNeighbours;
		std::vector<int> lowDimNeighbours;

		std::vector<weighted_tsne::scalar_type> distances_squared;
		hdi::dr::HDJointProbabilityGenerator<weighted_tsne::scalar_type>::Parameters temp_prob_gen_param;
		temp_prob_gen_param._perplexity = k;
		temp_prob_gen_param._perplexity_multiplier = 1;

		// computeHighDimensionalDistances includes the point itself as its nearest neighbour
		wt->prob_gen.computeHighDimensionalDistances(wt->data.data(), input_dims, N, distances_squared, highDimNeighbours, temp_prob_gen_param);
		wt->prob_gen.computeHighDimensionalDistances(wt->embedding.getContainer().data(), output_dims, N, distances_squared, lowDimNeighbours, temp_prob_gen_param);

		std::vector<float> errors;
		wt->calculate_set_error(lowDimNeighbours, highDimNeighbours, errors, N, (k + 1)); // d is dimensionality of the neighbourhoods which is (k+1) since it includes the point itself

		// Save errors as CSV
		save_as_csv(errors, N, 1, "C:/Users/basti/Google Drive/Learning/Master Thesis/ThesisDatasets/Generated/errors.csv");
	}

	delete wt;
}

int main() {
	//omp_set_num_threads(3);
	test_create_embedding();
	//test_jaccard_similarity();
	system("pause");
}