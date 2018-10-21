#ifndef OO_DNN_H
#define OO_DNN_H
#include<stdio.h>
#include<stdlib.h>
#include<random>
#include<assert.h>
#include<iostream>
#include<vector>
#include<math.h>
#include<string.h>
#include<fnmatch.h>
#include "OO_DNN.h"

using std::vector;
using std::cout;
using std::endl;

using namespace std;


class OO_DNN
{
    public:
        OO_DNN();
        virtual ~OO_DNN();
        std::vector<uint8_t> init_weights(int rows, int cols);
        std::vector<uint8_t> init_weights_bias(int rows, int cols);
        std::vector<uint8_t> mod_output(vector<uint8_t> x);
        int ind_identifier(vector<uint8_t> y);
        std::vector<uint8_t> subtract( const std::vector<uint8_t> &v1,const std::vector<uint8_t> &v2);
        std::vector<uint8_t> add( const std::vector<uint8_t> &v1, const std::vector<uint8_t> &v2);
        std::vector<uint8_t> multiply( const std::vector<uint8_t> &v1, const std::vector<uint8_t> &v2);
        std::vector<uint8_t> sigmoid(const std::vector<uint8_t>& v1);
        std::vector <uint8_t> sigmoid_d (const std::vector <uint8_t>& m1);
        std::vector <uint8_t> dot (const std::vector <uint8_t>& m1, const std::vector <uint8_t>& m2,const int m1_rows, const int m1_columns, const int m2_columns);
        std::vector<uint8_t> transpose (uint8_t *m, const int C, const int R) ;
        void print ( const vector <uint8_t>& v1, int v1_rows, int v1_columns );
        std::vector<uint8_t> getlabelVector(int n);
        vector<uint8_t> update_wandb (vector <uint8_t> &a , vector <uint8_t> &b, int num);
        void reinit (vector<uint8_t> &v);
        void train_model();
        std::vector<float> getErrorVec();
        void test();
        std::vector< std::vector <float> > getEvaluationMat();
        void print_vectors(std::vector< std::vector<float> > pvc);

    protected:

    private:
        const int n1;// Layer 1 (Input layer) neuron count
		const int n2;// Layer 2 (Hidden layer) neuron count
		const int n3;// Layer 3 (Output layer) neuron count
		 int l2_rows;// count of layer 2 rows
		 int l2_cols;// count of layer 2 columns
		 int l3_rows ;// count of layer 3 rows
		 int l3_cols ;// count of layer 3 columns
		 int epoch ;// Epoch count

		std::vector<uint8_t>  L2W;// Layer 2 Weights vector
		std::vector<uint8_t> L3W;// Layer 3 Weights vector
		std::vector<uint8_t> L2B;// Layer 2 Bias vector
		std::vector<uint8_t> L3B;// Layer 3 Bias vector

		std::vector<uint8_t> A2,A3;// Activation Vectors for hidden and output layers
		std::vector<uint8_t> A1;// Activation vector for input layer

		std::vector<uint8_t> z2,z3;// z is product of weights and input(previous activations) and sum of bias

		std::vector<uint8_t> d2,d3;// delta error for 2 and 3 layers

		std::vector<uint8_t> L2nw;// Layer 2 Weights vector
		std::vector<uint8_t> L3nw;// Layer 3 Weights vector
		std::vector<uint8_t> L2nb;// Layer 2 Bias vector
		std::vector<uint8_t> L3nb;// Layer 3 Bias vector
		std::vector< std::vector <float> > evaluation_mat;// 10 * 10 matrix for storage of confusion matrix
		// Total number of test dataset

        vector<float> error_mat;// end d3 error for each epoch

};

#endif // OO_DNN_H
