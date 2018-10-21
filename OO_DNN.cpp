#include "OO_DNN.h"
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



OO_DNN::OO_DNN()

   : n1(784),// Layer 1 (Input layer) neuron count
	 n2 (30),// Layer 2 (Hidden layer) neuron count
	 n3 (10)// Layer 3 (Output layer) neuron count
{
	 l2_rows =n2;// count of layer 2 rows
	 l2_cols =n1;// count of layer 2 columns
	 l3_rows =n3;// count of layer 3 rows
	 l3_cols =n2;// count of layer 3 columns
	 epoch=10;
    A1.reserve(784);
    L2nw.reserve(n2*n1);// Layer 2 Weights vector
    L3nw.reserve(n3*n2);// Layer 3 Weights vector
    L2nb.reserve(n2);// Layer 2 Bias vector
    L3nb.reserve(n3);// Layer 3 Bias vector
    evaluation_mat.resize(n3);
    error_mat.resize(epoch);
}

OO_DNN::~OO_DNN()
{
    //dtor
}

using namespace std;


                std::vector<uint8_t> OO_DNN::init_weights(int rows, int cols) {
                /*
                *Initialialising weights using normal distribution between the range of 0,1
                */
                    std::default_random_engine generator;
                    std::uniform_int_distribution<uint8_t> distribution(0,5);
                      // generates number in the range 1..6
                    std::vector<uint8_t> temp(rows*cols);

                    for (int i = 0; i < temp.size(); ++i) {

                            temp[i]=distribution(generator);
                            //temp[i] = get_random();
                    }

                    return temp;
                }

                std::vector<uint8_t> OO_DNN::init_weights_bias(int rows, int cols) {
                /*
                * Initialise Biases using normal distribution
                */
//                    unsigned seed = 3;
                   std::default_random_engine generator;

                    std::uniform_int_distribution<uint8_t> distribution(0,5);

                    std::vector<uint8_t> temp(rows*cols);

                    for (int i = 0; i < temp.size(); ++i) {

                            temp[i]=distribution(generator);
                            //temp[i] = get_random();
                    }

                    return temp;
                }



            std::vector<uint8_t> OO_DNN::mod_output(vector<uint8_t> x) {
                /*
                *	Returns vector of 10 elements with 1 on the class label
                */

                uint8_t temp_max;
                int location = 0;
                temp_max = x[0];

                for (int i = 1; i < x.size(); ++i) {
                    if (x[i] > temp_max) {
                        temp_max = x[i];
                        location = i ;
                    }
                }
                vector<uint8_t> ret = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


                ret[location] = 1.0;
                return ret;
            }


            int OO_DNN::ind_identifier(vector<uint8_t> y) {
            /*
            *returns integer as a identifier for class label during testing
            */
                uint8_t max;
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


             std::vector<uint8_t> OO_DNN::subtract( const std::vector<uint8_t> &v1, const std::vector<uint8_t> &v2){

            // Operator overload for vector subtraction

                 long int vec_size = v2.size();
                 std::vector<uint8_t> res(vec_size);

                 for(int i=0;i<vec_size;i++){
                     res[i] = v1[i] - v2[i];

                 }
                     return res;
             }


 std::vector<uint8_t> OO_DNN::add(const std::vector<uint8_t>& v1, const std::vector<uint8_t>& v2){


 // Operator overload for vector addition


	 long int vec_size = v1.size();
	 std::vector<uint8_t> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i]+v2[i];
	 }

	 	 return res;
 }



 std::vector<uint8_t> OO_DNN::multiply(const std::vector<uint8_t>& v1, const std::vector<uint8_t>& v2){

 	// Operator overload for vector multiplication

	 long int vec_size = v1.size();
	 std::vector<uint8_t> res(vec_size);

	 for(int i=0;i<vec_size;i++){
		 res[i] = v1[i]*v2[i];
	 }

	 return res;
}



 std::vector<uint8_t> OO_DNN::sigmoid(const std::vector<uint8_t>& v1){

 	// Gives activation for each vector
 	// Using Sigmoid Activation

	 long int vec_size = v1.size();
	 std::vector<uint8_t> res(vec_size);

	 for(unsigned i=0;i<vec_size;i++){

		 res[i]= (1 / (1 + expf(-v1[i]))) > 0.5 ? 1 : 0   ;   // defines the sigmoid function
	 }
	 return res;
}


 std::vector <uint8_t> OO_DNN::sigmoid_d (const std::vector <uint8_t>& m1) {

 	/*
 	* Function for derivative of sigmoid function
 	*/

     const unsigned long VECTOR_SIZE = m1.size();
     std::vector <uint8_t> output (VECTOR_SIZE);
	vector <uint8_t> temp (VECTOR_SIZE,1);


	output = multiply(sigmoid(m1),subtract(temp,sigmoid(m1)));
     return output;
 }


 std::vector <uint8_t> OO_DNN::dot (const std::vector <uint8_t>& m1, const std::vector <uint8_t>& m2,
                     const int m1_rows, const int m1_columns, const int m2_columns) {

 	/*
 	* Dot product between 2 matrices
 	*/

	 std::vector <uint8_t> output (m1_rows*m2_columns);

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


 std::vector<uint8_t> OO_DNN::transpose (uint8_t *m, const int C, const int R) {

 	/*
 	*	Transpose of Matrix
 	*/

     std::vector<uint8_t> mT (C*R);

     for(int n = 0; n!=C*R; n++) {
         int i = n/C;
         int j = n%C;
         mT[n] = m[R*j + i];
     }

     return mT;
 }


 void OO_DNN::print ( const vector <uint8_t>& v1, int v1_rows, int v1_columns ) {

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


 std::vector<uint8_t> OO_DNN::getlabelVector(int n){

 	/*
 	* Get label vector of image with an input
 	*/
	 std::vector<uint8_t> res = {0,0,0,0,0,0,0,0,0,0};

	 res[n]=1;

	 return res;
 }

 vector<uint8_t> OO_DNN::update_wandb (vector <uint8_t> &a , vector <uint8_t> &b, int num){

 	/*
 	*	Update the weights and biases matrices of all network layers after every mini batch
 	*
 	*/
	 float eta = 0.5;
	 int mini_batch_size = 10;
	vector <uint8_t> res (num);

 	//a = a - ((eta/mini_batch_size) * b);
	 for (int i = 0 ; i < res.size();++i){
 		b[i] = (eta/mini_batch_size) * b[i];
		res[i] = a[i] - b[i];
	 }
	//cout << endl << endl << " REACHED HERE###################" << endl << endl;
	return res;

 	}


void OO_DNN::reinit (vector<uint8_t> &v){

	/*
	*	Re initialise the delta vectors after every mini batches
	*/
	for (int i = 0;i < v.size();++i){
		v[i] = 0.0;
	}

}


void OO_DNN::print_vectors(std::vector< std::vector<float> > pvc) {

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

 void OO_DNN::train_model(){


        FILE *fp;
        char filename1[] ="/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/mnist_train.csv";// training input file link
//        char filename2[] ="/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/mnist_test.csv";// testing input file link
        FILE *write_out ;// file pointer for output file
        write_out = fopen("/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/training-reportdaa.txt","w");// file link to write the output of model and testing results

                                    // Layer 3 (Output layer) neuron coun
        char buff[2000];// buffer for reading from the csv file
        uint8_t activation[785];// input buffer holding vector
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
										activation[i] = atoi(token);
										 // can use atof to convert to float
										token = strtok(NULL,",");
										i++;
									}
									std::vector<uint8_t> label = getlabelVector(activation[0]);
							// feed forward
									for (int j=1; j<=A1.size();j++ ){
										if (activation[j] == 0){
											A1[j-1]= 0;
										}
										else{
											A1[j -1]= 1.0;
										}

										}//A1[j]=activation[j+1];
									z2=add(dot(L2W,A1,n2,n1,1),L2B);

									A2 = sigmoid(z2);
									z3=add(dot(L3W,A2,n3,n2,1),L3B);

									A3 = sigmoid(z3);
													// back propagation

									d3 = multiply(subtract(A3,label),sigmoid_d(z3));
									L3nb = add(d3,L3nb);
									L3nw = add(dot(d3,transpose(&A2[0],1,n2),n3,1,n2),L3nw); // gradient discent

									d2 = multiply(dot(transpose(&L3W[0],n2,n3),d3,n2,n3,1),sigmoid_d(z2));
									L2nb = add(d2,L2nb);
									L2nw = add(dot(d2,transpose(&A1[0],1,n1),n2,1,n1), L2nw) ; //gradient discent

							}
							else{
								//printf("weights updated")
								L3W = update_wandb(L3W,L3nw,n2*n3);
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

 }


 std::vector<float> OO_DNN::getErrorVec(){
 return error_mat;
 }


 void OO_DNN::test(){

        FILE *fp;
        //char filename1[] ="/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/mnist_train.csv";// training input file link
        char filename2[] ="/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/mnist_test.csv";// testing input file link
        FILE *write_out2 ;// file pointer for output file
        write_out2 = fopen("/home/puneetsingh/DIC_DNN/parallelized-NeuralNet/training-reportdaa.txt","w");// file link to write the output of model and testing results
      //  int tot_test_set = 10000;
    char buff[2000];// buffer for reading from the csv file
	uint8_t activation[785];// input buffer holding vector
	unsigned i=0;
	char * token;
	//int lab=0, batch_size=0;

		if ( (fp = fopen(filename2, "r") ) == NULL)    //  READ TEST FILE AND TEST MODEL ON THE DATA
				{
					printf("Cannot open %s.\n", filename2);
						  //  result = FAIL;
				}
				else{

					printf("File opened; ready to read.\n");
					for (int r = 0; r < evaluation_mat.size(); ++r) {
						evaluation_mat[r].resize(n3);
					}
					vector<uint8_t> pred_out;
					std::vector<uint8_t> label;
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
								activation[i]= atoi(token); // can use atof to convert to float
								token = strtok(NULL, ",");
								i++;
							}
							//printf("value of i:%d\n",i);
						 }
					//cout << "THe value of label is :" << activation[0] << endl;
						label = getlabelVector(activation[0]);//
							// feed forward
							cout<<"print label";
                        print(label,10,1);
						for (int j=1; j<=A1.size();j++ ){
							if (activation[j] == 0){
								A1[j-1]= 0;
							}
							else{
								A1[j-1]= 1;
							}
						}
						//A1[j]=activation[j+1];
						//printf("Printing A1:\n");
						//print(A1,784,1);
						z2= add(dot(L2W,A1,n2,n1,1),L2B);
						A2 = sigmoid(z2);
						//print(A2,15,1);

						z3=add(dot(L3W,A2,n3,n2,1),L3B);
						A3 = sigmoid(z3);


						pred_out = mod_output(A3);

						act_ind = ind_identifier(label);
						cout << "act_ind : " << act_ind << endl;

						pred_ind = ind_identifier(pred_out);
						cout << "pred_ind: " << pred_ind << endl;
						//print(A3,n3,1);
						evaluation_mat[act_ind][pred_ind] += 1.0;
						cout<<"\nprint results\n";

					}

				}

		cout << "Printing Evaluation Matrix:" << endl;
	//	print_vectors(evaluation_mat);
		fprintf(write_out2,"\n\n");
		fprintf(write_out2, "			EVALUATION MATRIX\n\n");
	for (int g = 0; g < evaluation_mat.size(); ++g) {
		for (auto y = evaluation_mat[g].begin(); y != evaluation_mat[g].end(); ++y) {
			fprintf(write_out2,"%.2f\t", *y);
		}
		fprintf(write_out2,"\n");
	}
	fprintf(write_out2, "\n");

    }


    std::vector<vector <float> > OO_DNN::getEvaluationMat(){

        return evaluation_mat;
    }






