#include "pch.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <immintrin.h>
#include <vector>
#include <thread>
#include <mutex>

// $CXX -O3 -mavx matmul-assignment.cpp

#if (!defined(_MSC_VER))
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#define SZ (1 << 2) // (1 << 10) == 1024

struct matFloat {
	float *data;
	const size_t sz;
	bool operator==(const matFloat &rhs) const {
		return !std::memcmp(data, rhs.data, sz*sz * sizeof(data[0]));
	}
};
struct matDouble {
	double *data;
	const size_t sz;
	bool operator==(const matDouble &rhs) const {
		return !std::memcmp(data, rhs.data, sz*sz * sizeof(data[0]));
	}
};

void matmul(matFloat &mres, const matFloat &m1, const matFloat &m2)
{
	for (int i = 0; i < mres.sz; i++) {
		for (int j = 0; j < mres.sz; j++) {
			mres.data[i*mres.sz + j] = 0;
			for (int k = 0; k < mres.sz; k++) {
				mres.data[i*mres.sz + j] += m1.data[i*mres.sz + k] * m2.data[k*mres.sz + j];
			}
		}
	}
}
void matmul(matDouble &mres, const matDouble &m1, const matDouble &m2)
{
	for (int i = 0; i < mres.sz; i++) {
		for (int j = 0; j < mres.sz; j++) {
			mres.data[i*mres.sz + j] = 0;
			for (int k = 0; k < mres.sz; k++) {
				mres.data[i*mres.sz + j] += m1.data[i*mres.sz + k] * m2.data[k*mres.sz + j];
			}
		}
	}
}

//SIMD Reduction
float simd_vreduce(const float *a, const size_t size)
{
	__m128 vsum = _mm_set1_ps(0.0f);
	for (std::size_t i = 0; i < size; i += 4) {
		__m128 v = _mm_load_ps(&a[i]);
		vsum = _mm_add_ps(vsum, v);
	}
	vsum = _mm_hadd_ps(vsum, vsum);
	vsum = _mm_hadd_ps(vsum, vsum);
	return _mm_cvtss_f32(vsum);
}
double simd_vreduce(const double *a, const size_t size) {
	__m256d vsum = _mm256_set1_pd(0);
	for (std::size_t i = 0; i < size; i += 4) {
		__m256d v = _mm256_load_pd(&a[i]);
		vsum = _mm256_add_pd(vsum, v);
	}
	vsum = _mm256_hadd_pd(vsum, vsum);
	__m256d vsump = _mm256_permute2f128_pd(vsum, vsum, 0x1);
	vsum = _mm256_add_pd(vsum, vsump);
	return _mm256_cvtsd_f64(vsum);
}

template <typename T>
inline void simd_matmul(T &mres, const T &m1, const T &m2, std::size_t N)
{
	alignas(32) float arrFloat[1024];//enough space for everthing up to and including biggest possible size 
	alignas(32) float colArrayFloat[1024];//enough space for everthing up to and including biggest possible size
	alignas(64) double arrDouble[1024];//enough space for everthing up to and including biggest possible size 
	alignas(64) double colArrayDouble[1024];//enough space for everthing up to and including biggest possible size

	if constexpr (std::is_same_v<T, matFloat>) {

		for (std::size_t j = 0; j < N; j++) {//m2

			for (std::size_t k = 0; k < N; k++) {//Get column from m2 into array
					colArrayFloat[k] = m2.data[(k*m2.sz) + j];//+j here moves to next column on consecutive iterations
			}

			for (std::size_t i = 0; i < N; i++) {//m1

				for (std::size_t m = 0; m < N; m += 4) {//current row of m1 * current column of m2
					__m128 row = _mm_load_ps(&m1.data[(i*m1.sz) + m]);
					__m128 col = _mm_load_ps(&colArrayFloat[m]);// load block of data from the array containing column data into a __m128
					__m128 mulRes = _mm_mul_ps(row, col);// do computation with row x col 
					_mm_store_ps(&arrFloat[m], mulRes);
				}
				float ReduceRes = simd_vreduce(arrFloat, N);//reduce mulRes to a singular value
				mres.data[i*N + j] = ReduceRes;
			}
		}
	}
	else if constexpr (std::is_same_v<T, matDouble>) {

		for (std::size_t j = 0; j < N; j++) {//m1

			for (std::size_t k = 0; k < N; k++) {//Get column from m2 into array
					colArrayDouble[k] = m2.data[(k*m2.sz) + j];//+i here moves to next column on consecutive iterations
			}

			for (std::size_t i = 0; i < N; i++) {//m2

				for (std::size_t m = 0; m < N; m += 4) {//current row of m1 * current column of m2
					__m256d row = _mm256_load_pd(&m1.data[(i*m1.sz) + m]);
					__m256d col = _mm256_load_pd(&colArrayDouble[m]);// load block of data from the array containing column data into a __m128
					__m256d mulRes = _mm256_mul_pd(row, col);// do computation with row x col 
					_mm256_store_pd(&arrDouble[m], mulRes);
				}
				float ReduceRes = simd_vreduce(arrDouble, N);//reduce mulRes to a singular value
				mres.data[i*N + j] = ReduceRes;
			}
		}
	}
}

template <typename T>
inline void simd_matmul_threads(T &mres, const T &m1, const T &m2, std::size_t N, int start, int limit)
{
	alignas(32) float arrFloatThreads[1024];//enough space for everthing up to and including biggest possible size 
	alignas(32) float colArrayFloatThreads[1024];//enough space for everthing up to and including biggest possible size
	alignas(64) double arrDoubleThreads[1024];//enough space for everthing up to and including biggest possible size 
	alignas(64) double colArrayDoubleThreads[1024];//enough space for everthing up to and including biggest possible size

	if constexpr (std::is_same_v<T, matFloat>) {

		for (std::size_t j = 0; j < N; j++) {//m1 - Calculate some rows against all columns

			for (std::size_t k = 0; k < N; k++) {//Get column from m2 into array
					colArrayFloatThreads[k] = m2.data[(k*m2.sz) + j];//+i here moves to next column on consecutive iterations
			}

			for (std::size_t i = 0; i < limit; i++) {//m2

				for (std::size_t m = 0; m < N; m += 4) {//current row of m1 * current column of m2
					__m128 row = _mm_load_ps(&m1.data[((i*m1.sz) + m) + start]);
					__m128 col = _mm_load_ps(&colArrayFloatThreads[m]);// load block of data from the array containing column data into a __m128
					__m128 mulRes = _mm_mul_ps(row, col);// do computation with row x col 
					_mm_store_ps(&arrFloatThreads[m], mulRes);
				}
				float ReduceRes = simd_vreduce(arrFloatThreads, N);//reduce mulRes to a singular value
				mres.data[(i*N + j) + start] = ReduceRes;
			}
		}
	}
	else if constexpr (std::is_same_v<T, matDouble>) {

		for (std::size_t j = 0; j < N; j++) {//m1

			for (std::size_t k = 0; k < N; k++) {//Get column from m2 into array
					colArrayDoubleThreads[k] = m2.data[(k*m2.sz) + j];//+i here moves to next column on consecutive iterations
			}

			for (std::size_t i = 0; i < limit; i++) {//m2

				for (std::size_t m = 0; m < N; m += 4) {//current row of m1 * current column of m2
					__m256d row = _mm256_load_pd(&m1.data[((i*m1.sz) + m) + start]);
					__m256d col = _mm256_load_pd(&colArrayDoubleThreads[m]);// load block of data from the array containing column data into a __m128
					__m256d mulRes = _mm256_mul_pd(row, col);// do computation with row x col 
					_mm256_store_pd(&arrDoubleThreads[m], mulRes);
				}
				float ReduceRes = simd_vreduce(arrDoubleThreads, N);//reduce mulRes to a singular value
				mres.data[(i*N + j) + start] = ReduceRes;
			}
		}
	}
}

template <typename T>
inline void simpleThreadFunction(T &mres, const T &m1, const T &m2, std::size_t N, int limit) {
	std::vector<std::thread> threads(4);
	for (unsigned i = 0; i < 4; i++){
		threads[i] = std::thread([i, &N, &mres, &m1, &m2, &limit] {
			if(i==0)//First thread
				simd_matmul_threads(mres, m1, m2, mres.sz, (N * limit) * 0, limit);
			if(i==1)//Seccond thread
				simd_matmul_threads(mres, m1, m2, mres.sz, (N * limit) * 1, limit);
			if(i==2)//Third thread
				simd_matmul_threads(mres, m1, m2, mres.sz, (N * limit) * 2, limit);
			if(i==3)//Fourth thread
				simd_matmul_threads(mres, m1, m2, mres.sz, (N * limit) * 3, limit);
		});
	}
	for (auto& t : threads) {
		t.join();
	}
}

void print_mat(const matFloat &m) {
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			std::cout << std::setw(3) << m.data[i*m.sz + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}
void print_mat(const matDouble &m) {
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			std::cout << std::setw(3) << m.data[i*m.sz + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

// A simply initialisation pattern. For a 4x4 matrix:

// 1   2  3  4
// 5   6  7  8
// 9  10 11 12
// 13 14 15 16

void init_mat(matFloat &m) {//1-N
	int count = 1;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = count++;
		}
	}
}
void init_mat(matDouble &m) {//1-N
	int count = 1;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = count++;
		}
	}
}
void init_mat_new(matFloat &m) {//1-9
	int count = 1;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = count++;
			if (count >= 10)
				count = 1;
		}
	}
}
void init_mat_new(matDouble &m) {//1-9
	int count = 1;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = count++;
			if (count >= 10)
				count = 1;
		}
	}
}

// Creates an identity matrix. For a 4x4 matrix:

// 1 0 0 0
// 0 1 0 0
// 0 0 1 0
// 0 0 0 1

void identity_mat(matFloat &m) {
	int count = 0;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = (count++ % (m.sz + 1)) ? 0 : 1;
		}
	}
}
void identity_mat(matDouble &m) {
	int count = 0;
	for (int i = 0; i < m.sz; i++) {
		for (int j = 0; j < m.sz; j++) {
			m.data[i*m.sz + j] = (count++ % (m.sz + 1)) ? 0 : 1;
		}
	}
}



int main(int argc, char *argv[])
{
	//-----------------------------------------------------------------------------------***_U/I_***
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::size_t N;
	std::cout << "Please enter matrix order: ";
	std::cin >> N;
	std::cout << "You have chosen a matrix of: " << N << "x" << N << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	//-----------------------------------------------------------------------------------***_CREATE THE MATRICES_***
	//FLOAT
	matFloat floatResDataSerial{ new float[N*N],N };
	matFloat floatResDataSIMD{ new float[N*N],N };
	matFloat floatResDataThreads{ new float[N*N],N };
	matFloat floatData{ new float[N*N],N };
	matFloat floatId{ new float[N*N],N };
	//DOUBLE
	matDouble doubleResDataSerial{ new double[N*N],N };
	matDouble doubleResDataSIMD{ new double[N*N],N };
	matDouble doubleResDataThreads{ new double[N*N],N };
	matDouble doubleData{ new double[N*N],N };
	matDouble doubleId{ new double[N*N],N };
	//-----------------------------------------------------------------------------------***_INITIALIZE AND PRINT THE MATRICES_***
	init_mat_new(floatData);
	init_mat_new(doubleData);
	/*std::cout << "float matrix:" << std::endl;
	print_mat(floatData);
	std::cout << "double matrix:" << std::endl;
	print_mat(doubleData);*/
	//-----------------------------------------------------------------------------------***_TIMING_***
	using namespace std::chrono;
	using tp_t = time_point<high_resolution_clock>;
	tp_t t1, t2;
	//-----------------------------------------------------------------------------------***_FLOAT CALCULATIONS_***
	//SERIAL
	t1 = high_resolution_clock::now();
	matmul(floatResDataSerial, floatData, floatData);
	t2 = high_resolution_clock::now();
	auto d = duration_cast<microseconds>(t2 - t1).count();
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "FLOAT calculations:" << std::endl;
	std::cout << std::endl;
	std::cout << "Serial:" << std::endl;
	std::cout << "Time taken to do matrix multiplication: " << d << " microsecconds." << std::endl;
	//SIMD
	std::cout << std::endl;
	std::cout << "SIMD:" << std::endl;
	t1 = high_resolution_clock::now();
	simd_matmul(floatResDataSIMD, floatData, floatData, floatResDataSIMD.sz);//SIMD mul
	t2 = high_resolution_clock::now();
	d = duration_cast<microseconds>(t2 - t1).count();
	std::cout << "Time taken to do matrix multiplication: " << d << " microsecconds." << std::endl;
	//Threads combined with SIMD
	int limit = N / 4;
	std::cout << std::endl;
	std::cout << "THREADS combined with SIMD:" << std::endl;
	t1 = high_resolution_clock::now();
	simpleThreadFunction(floatResDataThreads, floatData, floatData, floatResDataThreads.sz, limit);
	t2 = high_resolution_clock::now();
	d = duration_cast<microseconds>(t2 - t1).count();
	std::cout << "Time taken to do matrix multiplication: " << d << " microsecconds." << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	//-----------------------------------------------------------------------------------***_DOUBLE CALCULATIONS_***
	//SERIAL
	t1 = high_resolution_clock::now();
	matmul(doubleResDataSerial, doubleData, doubleData);
	t2 = high_resolution_clock::now();
	d = duration_cast<microseconds>(t2 - t1).count();
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "DOUBLE calculations:" << std::endl;
	std::cout << std::endl;
	std::cout << "Serial:" << std::endl;
	std::cout << "Time taken to do matrix multiplication: " << d << " microsecconds." << std::endl;
	std::cout << std::endl;
	//SIMD
	std::cout << "SIMD:" << std::endl;
	t1 = high_resolution_clock::now();
	simd_matmul(doubleResDataSIMD, doubleData, doubleData, doubleResDataSIMD.sz);//SIMD mul
	t2 = high_resolution_clock::now();
	d = duration_cast<microseconds>(t2 - t1).count();
	std::cout << "Time taken to do matrix multiplication: " << d << " microsecconds." << std::endl;
	//Threads combined with SIMD
	std::cout << std::endl;
	std::cout << "THREADS combined with SIMD:" << std::endl;
	t1 = high_resolution_clock::now();
	simpleThreadFunction(doubleResDataThreads, doubleData, doubleData, doubleResDataThreads.sz, limit);
	t2 = high_resolution_clock::now();
	d = duration_cast<microseconds>(t2 - t1).count();
	std::cout << "Time taken to do matrix multiplication: " << d << " microsecconds." << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	//-----------------------------------------------------------------------------------***_PRIT RESULT MATRICES_***
	//std::cout << "SERIAL FLOAT result:" << std::endl;
	//print_mat(floatResDataSerial);//print SERIAL mat mul result
	//std::cout << "SIMD FLOAT result:" << std::endl;
	//print_mat(floatResDataSIMD);//print SIMD mat mul result
	//std::cout << "SERIAL DOUBLE result:" << std::endl;
	//print_mat(doubleResDataSerial);//print SERIAL mat mul result
	//std::cout << "SIMD DOUBLE result:" << std::endl;
	//print_mat(doubleResDataSIMD);//print SIMD mat mul result
	/*std::cout << "SIMD FLOAT result:" << std::endl;
	print_mat(floatResDataSIMD);
	std::cout << "SIMD DOUBLE result:" << std::endl;
	print_mat(doubleResDataSIMD);
	std::cout << "Thread&SIMD FLOAT result:" << std::endl;
	print_mat(floatResDataThreads);
	std::cout << "Thread&SIMD DOUBLE result:" << std::endl;
	print_mat(doubleResDataThreads);*/
	//-----------------------------------------------------------------------------------***_TEST FOR EQUALITY_***
	const bool correctFloat = floatResDataSerial == floatResDataSIMD;
	const bool correctDouble = doubleResDataSerial == doubleResDataSIMD;
	const bool correctFloatThread = floatResDataSerial == floatResDataThreads;
	const bool correctDoubleThread = doubleResDataSerial == doubleResDataThreads;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "Check for equality:" << std::endl;
	std::cout << "Does serial result matrix equal the SIMD result matrix (FLOAT)?: " << correctFloat << std::endl;
	std::cout << "Does serial result matrix equal the SIMD result matrix (DOUBLE)?: " << correctDouble << std::endl;
	std::cout << "Does serial result matrix equal the Threads combined with SIMD result matrix (FLOAT)?: " << correctFloatThread << std::endl;
	std::cout << "Does serial result matrix equal the Threads combined with SIMD result matrix (DOUBLE)?: " << correctDoubleThread << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	//-----------------------------------------------------------------------------------***_THREADS (OLD, Thread stuff above this with the others now)***
	//unsigned num_cpus = std::thread::hardware_concurrency();
	//std::cout << "Launching " << num_cpus << " threads\n";
	//-----------------------------------------------------------------------------------***_CLEAN UP_***
	delete[] floatResDataSerial.data;
	delete[] floatResDataSIMD.data;
	delete[] floatData.data;
	delete[] floatId.data;
	delete[] doubleResDataSerial.data;
	delete[] doubleResDataSIMD.data;
	delete[] doubleData.data;
	delete[] doubleId.data;
	delete[] floatResDataThreads.data;
	delete[] doubleResDataThreads.data;
	//delete[] arrFloat;
	//delete[] colArrayFloat;
	//delete[] arrDouble;
	//delete[] colArrayDouble;

	system("pause");
	return correctFloat ? 0 : -1;
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
