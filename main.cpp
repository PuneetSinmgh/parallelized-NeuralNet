

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



 std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2){

// Operator overload for vector subtraction

	 long int vec_size = v1.size();
	 std::vector<float> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i] - v2[i];

	 }
	 	 return res;
 }



 std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2){


 // Operator overload for vector addition


	 long int vec_size = v1.size();
	 std::vector<float> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i]+v2[i];
	 }

	 	 return res;
 }


 std::vector<float> operator*(const std::vector<float>& v1, const std::vector<float>& v2){

 	// Operator overload for vector multiplication

	 long int vec_size = v1.size();
	 std::vector<float> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i]*v2[i];
	 }

	 return res;
}



vector<float> vec_division(vector<float> a, vector<float> b) {

	// Vector division function

	vector<float> c(a.size());

	for (int i = 0; i < a.size(); ++i) {
		c[i] = a[i] / b[i];
	}
	return c;
}


 void print ( const vector <float>& v1, int v1_rows, int v1_columns ) {

 	/*
 	* Print function for 1-D representation of a matrix
 	*/
	 for( int i = 0; i != v1_rows; ++i ) {
         for( int j = 0; j != v1_columns; ++j ) {
             cout << v1[ i * v1_columns + j ] << " ";
         }
         cout << '\n';
     }
     cout << endl;
 }

void print_vectors(std::vector<vector<float> > pvc) {

	/*
	* Print function for 2-D representation of matrix
	*/

	for (int g = 0; g < pvc.size(); ++g) {
		for (auto y = pvc[g].begin(); y != pvc[g].end(); ++y) {
			printf("%.2f\t", *y);
		}
		cout << endl;
	}

}

int main(){

        FILE *fp;
        char filename1[] ="/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/mnist_train.csv";// training input file link
        char filename2[] ="/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/mnist_test.csv";// testing input file link
        FILE *write_out ;// file pointer for output file
        write_out = fopen("/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/training-reportdaa.txt","w");// file link to write the output of model and testing results
        int tot_test_set = 10000;
        const int n1 = 784;// Layer 1 (Input layer) neuron count
		const int n2 = 30;// Layer 2 (Hidden layer) neuron count
		const int n3 = 10;// Layer 3 (Output layer) neuron coun
        std::vector< std::vector <float> > evaluation_mat;
        //evaluation_mat.reserve(n3);

    OO_DNN nn ;

    nn.train_model();

    nn.test();

    evaluation_mat = nn.getEvaluationMat();

    vector<float> TP(10);
	vector<float> FP(10);
	vector<float> FN(10);
	vector<float> TN(10);

	/*
	 * True Positive (TP) = diagonal elements of the evaluation matrix
	 */
	for (int e = 0; e < TP.size(); ++e) {
		TP[e] = evaluation_mat[e][e];
	}

	/*
	 * False Negative (FN) = sum of the column of the evaluation matrix for each class minus the diagonal element
	 */
	for (int f = 0; f < FN.size(); ++f) {
		FN[f] = 0.0;
		for (int i = 0; i < n3; ++i) {
			FN[f] += evaluation_mat[i][f];
		}
		FN[f] -= evaluation_mat[f][f];
	}

	/*
	 * False Positive (FP) = sum of the row elements of evaluation matrix for each class minus the diagonal element
	 */
	for (int t = 0; t < FP.size(); ++t) {
		FP[t] = 0.0;
		for (int i = 0; i < n3; ++i) {
			FP[t] += evaluation_mat[t][i];
		}
		FP[t] -= evaluation_mat[t][t];
	}

	for (int r = 0; r < TN.size(); ++r) {
		TN[r] = 0.0;

		TN[r] = tot_test_set - (TP[r] + FN[r] + FP[r]);
	}

	cout << "True Positive :" << endl;
	print(TP,1,n3);
	cout << endl;
	fprintf(write_out,"			TRUE POSITIVE:\n");
	for (int t = 0 ; t < TP.size();++t){
		fprintf(write_out,"%.0f\t",TP[t]);

	}
	fprintf(write_out,"\n\n");
	cout << "False Positive :" << endl;
	print(FP,1,n3);
	cout << endl;

	fprintf(write_out,"			FALSE POSITIVE:\n");
	for (int t = 0 ; t < FP.size();++t){
		fprintf(write_out,"%.0f\t",FP[t]);

	}
	fprintf(write_out,"\n\n");

	cout << "False Negative:" << endl;
	print(FN,1,n3);
	cout << endl;

	fprintf(write_out,"			FALSE NEGATIVE:\n");
	for (int t = 0 ; t < FN.size();++t){
		fprintf(write_out,"%.0f\t",FN[t]);

	}
	fprintf(write_out,"\n\n");

	cout << "True Negative :" << endl;
	print(TN,1,n3);
	cout << endl;

	fprintf(write_out,"			TRUE NEGATIVE:\n");
	for (int t = 0 ; t < TN.size();++t){
		fprintf(write_out,"%.0f\t",TN[t]);

	}
	fprintf(write_out,"\n\n");

	vector<float> precision(n3);
	vector<float> recall(n3);

	vector<float> accuracy(n3);

	precision = vec_division(TP, (TP + FP));
	recall = vec_division(TP,(TP + FN));
	accuracy = vec_division((TP + TN) , ((TP + FN) + (FP + TN)));

	float avg_precision = accumulate(precision.begin(), precision.end(), 0.0)
			/ precision.size();
	cout << "Printing Precision: " << avg_precision << endl;
	print(precision,1,n3);
	cout << endl;


	fprintf(write_out,"			PRECISION: %.3f\n",avg_precision);
	for (int t = 0 ; t < precision.size();++t){
		fprintf(write_out,"%.3f\t",precision[t]);

	}
	fprintf(write_out,"\n\n");

	float avg_recall = accumulate(recall.begin(), recall.end(), 0.0)
			/ recall.size();
	cout << "Printing Recall: " << avg_recall << endl;
	print(recall,1,n3);
	cout << endl;
	fprintf(write_out,"			RECALL: %.3f\n",avg_recall);
	for (int t = 0 ; t < recall.size();++t){
		fprintf(write_out,"%.3f\t",recall[t]);

	}
	fprintf(write_out,"\n\n");


	float avg_accuracy = accumulate(accuracy.begin(), accuracy.end(), 0.0)
			/ accuracy.size();
	cout << "Printing Accuracy: " << avg_accuracy << endl;
	print(accuracy,1,n3);
	cout << endl;
	fprintf(write_out,"			ACCURACY: %.3f\n",avg_accuracy);
	for (int t = 0 ; t < accuracy.size();++t){
		fprintf(write_out,"%.3f\t",accuracy[t]);

	}
	fprintf(write_out,"\n\n");



	fclose(write_out);


return 0;
}
