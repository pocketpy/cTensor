#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void assert_equal(Tensor a, Tensor b) {
	float epsilon = 1e-3;
	assert(a.data->numel == b.data->numel);

	cten_assert_shape("Shape not equal", a.shape, b.shape);

	for (int i = 0; i < a.data->numel; i++) {
		assert((a.data->flex[i] - b.data->flex[i] < epsilon) &&
			(a.data->flex[i] - b.data->flex[i] > -epsilon));
	}
}

Tensor create_tensor(TensorShape shape, float* data, bool requires_grad) {
	Tensor res = Tensor_new(shape, requires_grad);
	int numel = res.data->numel;
	for (int i = 0; i < numel; i++) {
		res.data->flex[i] = data[i];
	}
	return res;
}


int main() {
	float arange_data[] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
	   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
	   35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
	   52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
	   69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
	   86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96 };
	float ones_data[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	   1, 1, 1, 1, 1, 1, 1, 1 };

	//dim1 0*0
	Tensor a = create_tensor((TensorShape) { 0 }, arange_data, false);
	Tensor b = create_tensor((TensorShape) { 0 }, arange_data, false);
	Tensor c = Tensor_mul(a, b);
	float ans1[] = { 1 };
	Tensor ans = create_tensor((TensorShape) { 0 }, ans1, false);
	assert_equal(c, ans);

	//dim1 0*x
	a = create_tensor((TensorShape) { 0 }, arange_data, false);
	b = create_tensor((TensorShape) { 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans2[] = { 1,2,3,4 };
	ans = create_tensor((TensorShape) { 4 }, ans2, false);
	assert_equal(c, ans);

	//dim1 x*x
	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans3[] = { 1,4,9,16 };
	ans = create_tensor((TensorShape) { 4 }, ans3, false);
	assert_equal(c, ans);

	//dim2 x*x
	a = create_tensor((TensorShape) { 2, 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans4[] = { 1,4,9,16,25,36,49,64 };
	ans = create_tensor((TensorShape) { 2, 4 }, ans4, false);
	assert_equal(c, ans);

	//dim3 x*x
	a = create_tensor((TensorShape) { 2, 3, 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans5[] = { 1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144, 169,
	   196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576 };
	ans = create_tensor((TensorShape) { 2, 3, 4 }, ans5, false);
	assert_equal(c, ans);

	//dim4 x*x
	a = create_tensor((TensorShape) { 2, 3, 4, 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans6[] = { 1,    4,    9,   16,   25,   36,   49,   64,   81,  100,  121,
		144,  169,  196,  225,  256,  289,  324,  361,  400,  441,  484,
		529,  576,  625,  676,  729,  784,  841,  900,  961, 1024, 1089,
	   1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936,
	   2025, 2116, 2209, 2304, 2401, 2500, 2601, 2704, 2809, 2916, 3025,
	   3136, 3249, 3364, 3481, 3600, 3721, 3844, 3969, 4096, 4225, 4356,
	   4489, 4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929,
	   6084, 6241, 6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569, 7744,
	   7921, 8100, 8281, 8464, 8649, 8836, 9025, 9216 };
	ans = create_tensor((TensorShape) { 2, 3, 4, 4 }, ans6, false);
	assert_equal(c, ans);

	//dim0*dim2
	a = create_tensor((TensorShape) { 0 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans7[] = { 1,2,3,4,5,6,7,8 };
	ans = create_tensor((TensorShape) { 2, 4 }, ans7, false);
	assert_equal(c, ans);

	//dim0*dim3
	a = create_tensor((TensorShape) { 0 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans8[] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
	   18, 19, 20, 21, 22, 23, 24 };
	ans = create_tensor((TensorShape) { 2, 3, 4 }, ans8, false);
	assert_equal(c, ans);

	//dim0*dim4
	a = create_tensor((TensorShape) { 0 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4,4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans9[] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
	   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
	   35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
	   52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
	   69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
	   86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans9, false);
	assert_equal(c, ans);

	//dim1*dim2
	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 3, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans10[] = { 1,  4,  9, 16,  5, 12, 21, 32,9,20,33,48 };
	ans = create_tensor((TensorShape) { 3, 4 }, ans10, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans11[] = { 1,  4,  9, 16 };
	ans = create_tensor((TensorShape) { 1, 4 }, ans11, false);
	assert_equal(c, ans);

	//dim1*dim3
	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans12[] = { 1,  4,  9, 16,  5, 12, 21, 32,  9, 20, 33, 48, 13, 28, 45, 64, 17,
	   36, 57, 80, 21, 44, 69, 96 };
	ans = create_tensor((TensorShape) { 2, 3, 4 }, ans12, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 3, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans13[] = { 1,  4,  9, 16,  5, 12, 21, 32,  9, 20, 33, 48 };
	ans = create_tensor((TensorShape) { 1, 3, 4 }, ans13, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 1, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans14[] = { 1,  4,  9, 16,  5, 12, 21, 32 };
	ans = create_tensor((TensorShape) { 2, 1, 4 }, ans14, false);
	assert_equal(c, ans);

	//dim1*dim4
	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4,4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans15[] = { 1,   4,   9,  16,   5,  12,  21,  32,   9,  20,  33,  48,  13,
		28,  45,  64,  17,  36,  57,  80,  21,  44,  69,  96,  25,  52,
		81, 112,  29,  60,  93, 128,  33,  68, 105, 144,  37,  76, 117,
	   160,  41,  84, 129, 176,  45,  92, 141, 192,  49, 100, 153, 208,
		53, 108, 165, 224,  57, 116, 177, 240,  61, 124, 189, 256,  65,
	   132, 201, 272,  69, 140, 213, 288,  73, 148, 225, 304,  77, 156,
	   237, 320,  81, 164, 249, 336,  85, 172, 261, 352,  89, 180, 273,
	   368,  93, 188, 285, 384 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans15, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 3, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans16[] = { 1,   4,   9,  16,   5,  12,  21,  32,   9,  20,  33,  48,  13,
		28,  45,  64,  17,  36,  57,  80,  21,  44,  69,  96,  25,  52,
		81, 112,  29,  60,  93, 128,  33,  68, 105, 144,  37,  76, 117,
	   160,  41,  84, 129, 176,  45,  92, 141, 192 };
	ans = create_tensor((TensorShape) { 1, 3, 4, 4 }, ans16, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 1, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans17[] = { 1,   4,   9,  16,   5,  12,  21,  32,   9,  20,  33,  48,  13,
		28,  45,  64,  17,  36,  57,  80,  21,  44,  69,  96,  25,  52,
		81, 112,  29,  60,  93, 128 };
	ans = create_tensor((TensorShape) { 2, 1, 4, 4 }, ans17, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 1, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans18[] = { 1,  4,  9, 16,  5, 12, 21, 32,  9, 20, 33, 48, 13, 28, 45, 64, 17,
	   36, 57, 80, 21, 44, 69, 96 };
	ans = create_tensor((TensorShape) { 2, 3, 1, 4 }, ans18, false);
	assert_equal(c, ans);

	//dim2*dim3
	a = create_tensor((TensorShape) { 3,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans19[] = { 1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144,  13,
		28,  45,  64,  85, 108, 133, 160, 189, 220, 253, 288 };
	ans = create_tensor((TensorShape) { 2, 3, 4 }, ans19, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 3,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 3, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans20[] = { 1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144 };
	ans = create_tensor((TensorShape) { 1, 3, 4 }, ans20, false);
	assert_equal(c, ans);

	//dim2*dim4
	a = create_tensor((TensorShape) { 4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans21[] = { 1,    4,    9,   16,   25,   36,   49,   64,   81,  100,  121,
		144,  169,  196,  225,  256,   17,   36,   57,   80,  105,  132,
		161,  192,  225,  260,  297,  336,  377,  420,  465,  512,   33,
		 68,  105,  144,  185,  228,  273,  320,  369,  420,  473,  528,
		585,  644,  705,  768,   49,  100,  153,  208,  265,  324,  385,
		448,  513,  580,  649,  720,  793,  868,  945, 1024,   65,  132,
		201,  272,  345,  420,  497,  576,  657,  740,  825,  912, 1001,
	   1092, 1185, 1280,   81,  164,  249,  336,  425,  516,  609,  704,
		801,  900, 1001, 1104, 1209, 1316, 1425, 1536 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans21, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 3, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans22[] = { 1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144, 169,
	   196, 225, 256,  17,  36,  57,  80, 105, 132, 161, 192, 225, 260,
	   297, 336, 377, 420, 465, 512,  33,  68, 105, 144, 185, 228, 273,
	   320, 369, 420, 473, 528, 585, 644, 705, 768 };
	ans = create_tensor((TensorShape) { 1, 3, 4,4 }, ans22, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 1, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans23[] = { 1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144, 169,
	   196, 225, 256,  17,  36,  57,  80, 105, 132, 161, 192, 225, 260,
	   297, 336, 377, 420, 465, 512 };
	ans = create_tensor((TensorShape) { 2, 1, 4,4 }, ans23, false);
	assert_equal(c, ans);

	//dim3*dim4
	a = create_tensor((TensorShape) { 3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 3, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans24[] = { 1,    4,    9,   16,   25,   36,   49,   64,   81,  100,  121,
		144,  169,  196,  225,  256,  289,  324,  361,  400,  441,  484,
		529,  576,  625,  676,  729,  784,  841,  900,  961, 1024, 1089,
	   1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936,
	   2025, 2116, 2209, 2304,   49,  100,  153,  208,  265,  324,  385,
		448,  513,  580,  649,  720,  793,  868,  945, 1024, 1105, 1188,
	   1273, 1360, 1449, 1540, 1633, 1728, 1825, 1924, 2025, 2128, 2233,
	   2340, 2449, 2560, 2673, 2788, 2905, 3024, 3145, 3268, 3393, 3520,
	   3649, 3780, 3913, 4048, 4185, 4324, 4465, 4608 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans24, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 3, 4, 4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans25[] = { 1,    4,    9,   16,   25,   36,   49,   64,   81,  100,  121,
		144,  169,  196,  225,  256,  289,  324,  361,  400,  441,  484,
		529,  576,  625,  676,  729,  784,  841,  900,  961, 1024, 1089,
	   1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936,
	   2025, 2116, 2209, 2304 };
	ans = create_tensor((TensorShape) { 1, 3, 4,4 }, ans25, false);
	assert_equal(c, ans);

	//special dim4*dim4
	a = create_tensor((TensorShape) { 2,3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 1,1,1 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans26[] = { 1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
		14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
		27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
		40,  41,  42,  43,  44,  45,  46,  47,  48,  98, 100, 102, 104,
	   106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130,
	   132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156,
	   158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182,
	   184, 186, 188, 190, 192 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans26, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 2,3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 3,1,1 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans27[] = { 1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
		14,  15,  16,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,
		54,  56,  58,  60,  62,  64,  99, 102, 105, 108, 111, 114, 117,
	   120, 123, 126, 129, 132, 135, 138, 141, 144,  49,  50,  51,  52,
		53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64, 130,
	   132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156,
	   158, 160, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273,
	   276, 279, 282, 285, 288 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans27, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 2,3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 1,4,1 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans28[] = { 1,   2,   3,   4,  10,  12,  14,  16,  27,  30,  33,  36,  52,
		56,  60,  64,  17,  18,  19,  20,  42,  44,  46,  48,  75,  78,
		81,  84, 116, 120, 124, 128,  33,  34,  35,  36,  74,  76,  78,
		80, 123, 126, 129, 132, 180, 184, 188, 192,  49,  50,  51,  52,
	   106, 108, 110, 112, 171, 174, 177, 180, 244, 248, 252, 256,  65,
		66,  67,  68, 138, 140, 142, 144, 219, 222, 225, 228, 308, 312,
	   316, 320,  81,  82,  83,  84, 170, 172, 174, 176, 267, 270, 273,
	   276, 372, 376, 380, 384 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans28, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 2,3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 1,4,1 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans29[] = { 1,   2,   3,   4,  10,  12,  14,  16,  27,  30,  33,  36,  52,
		56,  60,  64,  17,  18,  19,  20,  42,  44,  46,  48,  75,  78,
		81,  84, 116, 120, 124, 128,  33,  34,  35,  36,  74,  76,  78,
		80, 123, 126, 129, 132, 180, 184, 188, 192, 245, 250, 255, 260,
	   318, 324, 330, 336, 399, 406, 413, 420, 488, 496, 504, 512, 325,
	   330, 335, 340, 414, 420, 426, 432, 511, 518, 525, 532, 616, 624,
	   632, 640, 405, 410, 415, 420, 510, 516, 522, 528, 623, 630, 637,
	   644, 744, 752, 760, 768 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans29, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 2,3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 2, 1,1,4 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans30[] = { 1,   4,   9,  16,   5,  12,  21,  32,   9,  20,  33,  48,  13,
		28,  45,  64,  17,  36,  57,  80,  21,  44,  69,  96,  25,  52,
		81, 112,  29,  60,  93, 128,  33,  68, 105, 144,  37,  76, 117,
	   160,  41,  84, 129, 176,  45,  92, 141, 192, 245, 300, 357, 416,
	   265, 324, 385, 448, 285, 348, 413, 480, 305, 372, 441, 512, 325,
	   396, 469, 544, 345, 420, 497, 576, 365, 444, 525, 608, 385, 468,
	   553, 640, 405, 492, 581, 672, 425, 516, 609, 704, 445, 540, 637,
	   736, 465, 564, 665, 768 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans30, false);
	assert_equal(c, ans);

	a = create_tensor((TensorShape) { 2,3,4,4 }, arange_data, false);
	b = create_tensor((TensorShape) { 1, 3,4,1 }, arange_data, false);
	c = Tensor_mul(a, b);
	float ans31[] = { 1,    2,    3,    4,   10,   12,   14,   16,   27,   30,   33,
		 36,   52,   56,   60,   64,   85,   90,   95,  100,  126,  132,
		138,  144,  175,  182,  189,  196,  232,  240,  248,  256,  297,
		306,  315,  324,  370,  380,  390,  400,  451,  462,  473,  484,
		540,  552,  564,  576,   49,   50,   51,   52,  106,  108,  110,
		112,  171,  174,  177,  180,  244,  248,  252,  256,  325,  330,
		335,  340,  414,  420,  426,  432,  511,  518,  525,  532,  616,
		624,  632,  640,  729,  738,  747,  756,  850,  860,  870,  880,
		979,  990, 1001, 1012, 1116, 1128, 1140, 1152 };
	ans = create_tensor((TensorShape) { 2, 3, 4,4 }, ans31, false);
	assert_equal(c, ans);

	return 0;
}