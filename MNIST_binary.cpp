
#include<stdio.h>
#include<stdlib.h>
#include<random>
#include<assert.h>
#include<iostream>
#include<vector>
#include<math.h>
#include<string.h>
#include<fnmatch.h>

using std::vector;
using std::cout;
using std::endl;

using namespace std;


	std::vector<float> init_weights(int rows, int cols) {
	/*
	*Initialialising weights using normal distribution between the range of 0,1 
	*/	
	unsigned seed = 10;
	std::default_random_engine generator (seed);
		
  	std::normal_distribution<float> distribution (0.0,1.0);

		std::vector<float> temp(rows*cols);

		for (int i = 0; i < temp.size(); ++i) {

				temp[i]=distribution(generator)/28;
				//temp[i] = get_random();
		}

		return temp;
	}

	std::vector<float> init_weights_bias(int rows, int cols) {
	/*
	* Initialise Biases using normal distribution
	*/
		unsigned seed = 100;
		std::default_random_engine generator (seed);

  		std::normal_distribution<float> distribution (0.0,1.0);

		std::vector<float> temp(rows*cols);

		for (int i = 0; i < temp.size(); ++i) {

				temp[i]=distribution(generator);
				//temp[i] = get_random();
		}

		return temp;
	}

		const float n1 = 784;// Layer 1 (Input layer) neuron count
		const float n2 = 30;// Layer 2 (Hidden layer) neuron count
		const float n3 = 10;// Layer 3 (Output layer) neuron count
		const float l2_rows = n2;// count of layer 2 rows
		const float l2_cols = n1;// count of layer 2 columns
		const float l3_rows = n3;// count of layer 3 rows 
		const float l3_cols = n2;// count of layer 3 columns
		const int epoch = 3;// Epoch count

		std::vector<float>  L2W;// Layer 2 Weights vector
		std::vector<float> L3W;// Layer 3 Weights vector
		std::vector<float> L2B;// Layer 2 Bias vector
		std::vector<float> L3B;// Layer 3 Bias vector

		std::vector<float> A2,A3;// Activation Vectors for hidden and output layers
		std::vector<float> A1(n1);// Activation vector for input layer

		std::vector<float> z2,z3;// z is product of weights and input(previous activations) and sum of bias 

		std::vector<float> d2,d3;// delta error for 2 and 3 layers

		std::vector<float> L2nw(n2*n1,0.0);// Layer 2 Weights vector
		std::vector<float> L3nw(n3*n2,0.0);// Layer 3 Weights vector
		std::vector<float> L2nb(n2*1,0.0);// Layer 2 Bias vector
		std::vector<float> L3nb(n3*1,0.0);// Layer 3 Bias vector
		std::vector<vector <float> > evaluation_mat (n3);// 10 * 10 matrix for storage of confusion matrix
		int tot_test_set = 10000;// Total number of test dataset
		 
	vector<float> error_mat (epoch);// end d3 error for each epoch 


std::vector<float> mod_output(vector<float> x) {
	/*
	*	Returns vector of 10 elements with 1 on the class label 
	*/	
	
	float temp_max;
	int location = 0;
	temp_max = x[0];

	for (int i = 1; i < x.size(); ++i) {
		if (x[i] > temp_max) {
			temp_max = x[i];
			location = i ;
		}
	}
	vector<float> ret = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	
	ret[location] = 1.0;
	return ret;
}

int ind_identifier(vector<float> y) {
/*
*returns integer as a identifier for class label during testing
*/
	float max;
	int loc = 0;
	max = y[0];

	for (int i = 1; i < y.size(); ++i) {
		if (y[i] > max) {
			max = y[i];
			loc = i ;
			
		}
	}
	return loc ;

}


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


 std::vector<float> sigmoid(const std::vector<float>& v1){

 	// Gives activation for each vector 
 	// Using Sigmoid Activation

	 long int vec_size = v1.size();
	 std::vector<float> res(vec_size);

	 for(unsigned i=0;i<vec_size;i++){

		 res[i]= 1.0 / (1.0 + expf(-v1[i]));   // defines the sigmoid function
	 }

	 return res;
}


 std::vector <float> sigmoid_d (const std::vector <float>& m1) {

 	/*
 	* Function for derivative of sigmoid function 
 	*/

     const unsigned long VECTOR_SIZE = m1.size();
     std::vector <float> output (VECTOR_SIZE);
	vector <float> temp (VECTOR_SIZE,1);
	

	output = sigmoid(m1)*(temp - sigmoid(m1));
     return output;
 }


 std::vector <float> dot (const std::vector <float>& m1, const std::vector <float>& m2,
                     const int m1_rows, const int m1_columns, const int m2_columns) {

 	/*
 	* Dot product between 2 matrices
 	*/

	 std::vector <float> output (m1_rows*m2_columns);

     for( int row = 0; row != m1_rows; ++row ) {

         for( int col = 0; col != m2_columns; ++col ) {

             output[ row * m2_columns + col ] = 0.f;

             for(int k = 0; k != m1_columns; ++k ) {

                 output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
             }
         }
     }

     return output;
 }


 std::vector<float> transpose (float *m, const int C, const int R) {

 	/* 
 	*	Transpose of Matrix	
 	*/

     std::vector<float> mT (C*R);

     for(int n = 0; n!=C*R; n++) {
         int i = n/C;
         int j = n%C;
         mT[n] = m[R*j + i];
     }

     return mT;
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

void print_vectors(vector<vector<float> > pvc) {

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

 std::vector<float> getlabelVector(int n){

 	/* 
 	* Get label vector of image with an input
 	*/
	 std::vector<float> res = {0,0,0,0,0,0,0,0,0,0};

	 res[n]=1;

	 return res;
 }

 vector<float> update_wandb (vector <float> &a , vector <float> &b, int num){

 	/*
 	*	Update the weights and biases matrices of all network layers after every mini batch
 	*
 	*/
	 float eta = 0.5;
	 int mini_batch_size = 10;
	vector <float> res (num);

 	//a = a - ((eta/mini_batch_size) * b);
	 for (int i = 0 ; i < res.size();++i){
 		b[i] = (eta/mini_batch_size) * b[i];
		res[i] = a[i] - b[i];	
	 }
	//cout << endl << endl << " REACHED HERE###################" << endl << endl;	
	return res;
	
 	}


void reinit (vector<float> &v){
	
	/*
	*	Re initialise the delta vectors after every mini batches
	*/	
	for (int i = 0;i < v.size();++i){
		v[i] = 0.0;
	}

}

int main(){
	
	FILE *fp;
	char filename1[] = "/home/romal-patel/Desktop/mnist_train.csv";// training input file link
	char filename2[] = "/home/romal-patel/Desktop/mnist_test.csv";// testing input file link
	FILE *write_out ;// file pointer for output file
	write_out = fopen("/home/romal-patel/Desktop/training-reportdaa.txt","w");// file link to write the output of model and testing results
	char buff[2000];// buffer for reading from the csv file
	float activation[785];// input buffer holding vector
	unsigned i=0;
	char * token;	
	int lab=0, batch_size=0;

		/* 
		* INITIALISING ALL WEIGHTS AND BIASES OF THE NETWORK 
		*/

		L2W = init_weights(l2_rows, l2_cols);// initializing wieghts for layer 2
		
		cout << "L2W-" << endl;
		print(L2W,1,n2*n1);

		L2B = init_weights_bias( l2_cols,1); // initializing weights for layer 3
		/*
		 * Layer 3- 15 * 10 weights initialization vector
		 * 			1 * 10 Bias vector
		 */
		cout << "L2B-" << endl;
		print(L2B,1,n2);		
		L3W = init_weights(l3_rows, l3_cols);
		
		cout << "L3W-" << endl;
		print(L3W,1,n2*n3);
		L3B = init_weights_bias(l3_cols,1);
		cout << "L3B-" << endl;
		print(L3B,1,n3);

		/*
		*	TRAINING MODEL
		*/
		for (int z = 1;z <= epoch;++z){// FOR EACH EPOCH
			if ( (fp = fopen(filename1, "r") ) == NULL)
			{
				printf("Cannot open %s.\n", filename1);
			}
			else{
				while(!feof(fp)){
					i=0;
				
					if(fgets(buff, 2000 ,fp )!=NULL){

						token = strtok(buff,",");
						++batch_size;
							if(batch_size<=10){
								
									while(token!=NULL){
										activation[i] = atof(token);
										 // can use atof to convert to float
										token = strtok(NULL,",");
										i++;
									}
									std::vector<float> label = getlabelVector(activation[0]);
							// feed forward
									for (int j=1; j<=A1.size();j++ ){
										if (activation[j] == 0){
											A1[j-1]= 0;	
										}	
										else{
											A1[j -1]= 1.0;
										}
			
										}//A1[j]=activation[j+1];
									z2=dot(L2W,A1,n2,n1,1)+L2B;
				//printf("Printing z2:");					
				//print(z2,1,30);
									A2 = sigmoid(z2);
									z3=dot(L3W,A2,n3,n2,1)+L3B;
				//printf("Printing z3");					
				//print(z3,1,10);
									A3 = sigmoid(z3);
													// back propagation
									//d3 = (A3-label)*sigmoid_d(z3);
									d3 = (A3-label) * sigmoid_d(z3);	
									L3nb = d3 + L3nb;
									L3nw = dot(d3,transpose(&A2[0],1,n2),n3,1,n2) + L3nw; // gradient discent
									d2 = dot(transpose(&L3W[0],n2,n3),d3,n2,n3,1)*sigmoid_d(z2);
									L2nb = d2 + L2nb;
									L2nw = dot(d2,transpose(&A1[0],1,n1),n2,1,n1) + L2nw ; //gradient discent									
									//printf("D2:\n");
//print(d2,30,1);
//			printf("D3:\n");
//			print(d3,10,1);
					
							}
							else{
								//printf("weights updated")
			;					L3W = update_wandb(L3W,L3nw,n2*n3);
								L3B = update_wandb(L3B,L3nb,n3);
								L2W = update_wandb(L2W,L2nw,n1*n2);
								L2B = update_wandb(L2B,L2nb,n2);
								batch_size = 0;
								reinit(L3nw);
								reinit(L3nb);
								reinit(L2nw);
								reinit(L2nb);
								continue;
							}
					}
					//printf("reading line and learning\n");
				}
				printf("activations\n");
				print(A3,n3,1);
				
			}
		printf ("epoch:%d\n",z);
		
				//printf("Printing D3:\n");
				//print(d3,n3,1);
				error_mat[z-1] =  (float) accumulate(d3.begin(), d3.end(), 0.0) / d3.size();
		}
		cout << "Layer 2 W-" << endl;		
		print(L2W,1,n2*n1);
		cout << "Layer 2 B-" << endl;
		print(L2B,1,n2);

		cout << "Layer 3 W-" << endl;
		print(L3W,1,n3*n2);
		cout << "Layer 3 B-" << endl;
		print(L3B,1,n3);
		cout << "Printing Error over Epochs:" << endl;
		for (int u = 0; u < error_mat.size();++u){
			cout << "Epoch " << u+1 << ":" << error_mat[u] << endl;	
			fprintf(write_out,"Epoch %d: %f\n",u+1,error_mat[u]);
			}


			/*
			*	TESTING MODEL 
			*	every image from test file is fed forward through the network. The actual label and predicted label is recorded 
			*	and added to the evaluation matrix.
			*/

		if ( (fp = fopen(filename2, "r") ) == NULL)
				{
					printf("Cannot open %s.\n", filename2);
						  //  result = FAIL;
				}
				else{

					printf("File opened; ready to read.\n");
					for (int r = 0; r < evaluation_mat.size(); ++r) {
						evaluation_mat[r].resize(n3);
					}
					vector<float> pred_out;
					std::vector<float> label;
					int act_ind; // index of class label for actual output
					int pred_ind;// index of class label for predicted output 
					while(!feof(fp)){
						i=0;
						if(fgets(buff, 2000 ,fp )!=NULL){

							token = strtok(buff,",");
						//	printf("%s\n",token);

							//lab = atoi(token);

							while(token!=NULL){
								//printf("%s\n",token);
								activation[i]= atof(token); // can use atof to convert to float
								token = strtok(NULL, ",");
								i++;
							}
							//printf("value of i:%d\n",i);
						 }
					//cout << "THe value of label is :" << activation[0] << endl;
						label = getlabelVector(activation[0]);// 
							// feed forward
				
						for (int j=1; j<=A1.size();j++ ){
							if (activation[j] == 0){
								A1[j-1]= 0;	
							}	
							else{
								A1[j -1]= 1.0;
							}						
						}
						//A1[j]=activation[j+1];
						//printf("Printing A1:\n");						
						//print(A1,784,1);
						z2=dot(L2W,A1,n2,n1,1)+L2B;
						A2 = sigmoid(z2);
						//print(A2,15,1);

						z3=dot(L3W,A2,n3,n2,1)+L3B;
						A3 = sigmoid(z3);

	
						pred_out = mod_output(A3);
						
						act_ind = ind_identifier(label);
						//cout << "act_ind : " << act_ind << endl;
												
						pred_ind = ind_identifier(pred_out);
						//cout << "pred_ind: " << pred_ind << endl;						
						//print(A3,n3,1);	
						evaluation_mat[act_ind][pred_ind] += 1.0;

						
						//printf("\nprint results\n");
						
					}
						
				}

		cout << "Printing Evaluation Matrix:" << endl;
		print_vectors(evaluation_mat);
		fprintf(write_out,"\n\n");
		fprintf(write_out, "			EVALUATION MATRIX\n\n");
	for (int g = 0; g < evaluation_mat.size(); ++g) {
		for (auto y = evaluation_mat[g].begin(); y != evaluation_mat[g].end(); ++y) {
			fprintf(write_out,"%.2f\t", *y);
		}
		fprintf(write_out,"\n");
	}
	fprintf(write_out, "\n");


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
