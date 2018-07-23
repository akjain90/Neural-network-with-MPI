#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include<sys/time.h>
#include <string>
#include "neural_net.h"

using namespace std;
double wcTimeStart = 0.0 , wcTimeEnd = 0.0;
vector<double> J_train_vec;
vector<double> J_test_vec;

vector<double> X_train_ip;		//for input to complete training dataset to root process
vector<double> y_train_ip;		//for input to complete training dataset to root process	
vector<double> X_test_ip;		//for input to complete test dataset to root process
vector<double> y_test_ip;		//for input to complete test dataset to root process	
//X = features*samples
//vector<double> X{1,5,2,8,3,8,1,10,3,7,6,15,1,4,19,9,5,8,1,20,25,1,2,7,18,11,10,23,5,2};
vector<double> X_train;		//training data for the individual process after work distribution (80% data)
vector<double> X_test;		//training data for the individual process after work distribution (20% data)
vector<double> X_test1;
//y = samples*1;
//vector<double> y{1,1,2,2,3,3};
vector<double> y_train;		//training data for the individual process after work distribution (80% data)
vector<double> y_test;		//training data for the individual process after work distribution (20% data)
vector<double> y_test1;
//vector to store the predicted label
vector<double> test_label;
//h = labels*samples
vector<double> h_train;
vector<double> h_test;
vector<double> h_test1;
/* int input_nodes= 2025;
	int hidden_nodes = 50;
	int output_nodes = 10; */
double J_old_train = 0.0;		//network cost
double J_new_train = 0.0;		//network cost
double J_test = 0.0;
double t_k = 0.8;
double lambda = 0.0;
vector<double> lambda_vec;
vector<double> samples_vec;
int myrank;
int psize;
int Q=0,R=0;		//variables used for the work division 
					//Q = (total work / total MPI process),		R = (total work % total process)

int *X_sendcounts = NULL;
int *X_displ = NULL;
int *y_sendcounts = NULL;
int *y_displ = NULL;

/* int input_nodes= 5;
int hidden_nodes = 4;
int output_nodes = 3; */
int input_nodes= 400;
int hidden_nodes = 25;
int output_nodes = 10;
//int input_nodes= 2025;
//int hidden_nodes = 100;
//int output_nodes = 10;
int num_samples_train = 0;		//all process have this variable set and shows number of training samples per process
int num_samples_test = 0;		//all process have this variable set and shows number of test samples per process
int total_samples_train = 0;		//only processor 0 will have this variable set which shows total number of samples
int total_samples_test = 0;
int test_samples = 0;
int test_samples1 = 0;
neural_net input_layer(input_nodes, hidden_nodes);
neural_net hidden_layer(hidden_nodes,output_nodes);
neural_net output_layer(output_nodes);
vector<double> delta_3(output_nodes,0);
vector<double> delta_2(hidden_nodes,0);

double getSeconds()
{
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}


void feedforward(vector<double>::iterator X_train_0, vector<double>::iterator h_train_0, int &samples)
{
	for(int s=0;s<samples;s++)
	{
		input_layer.cala(X_train_0+s*input_nodes);
		hidden_layer.calz();
		hidden_layer.cala();
		output_layer.calz();
		output_layer.calhypothesis(h_train_0+s*output_nodes);
	}
	//cout<<"Hello i am rank "<<myrank<<endl;
}

///feedforward for predict function///
void feedforward(vector<double>::iterator X_test_0, vector<double>::iterator h_test_0,int &samples, int test)
{
	for(int s=0;s<samples;s++)
	{
		input_layer.cala(X_test_0+s*input_nodes);
		hidden_layer.calz();
		hidden_layer.cala();
		output_layer.calz();
		test_label.push_back(output_layer.calhypothesis(h_test_0));
	}
}

void backpropagation(vector<double>::iterator X_train_0, vector<double>::iterator h_train_0, int &samples)
{
 (hidden_layer.theta_grad_temp).resize(0);
 (hidden_layer.theta_grad_temp).resize(hidden_layer.resize_param,0);	
 (input_layer.theta_grad_temp).resize(0);
 (input_layer.theta_grad_temp).resize(input_layer.resize_param,0);
auto theta_2_0 = hidden_layer.theta.begin();		//starting address of theta_2
auto theta2_grad_0 = hidden_layer.theta_grad.begin();
 for(auto s = 0;s<samples;s++)
 {
	input_layer.cala(X_train_0+s*input_nodes);
	hidden_layer.calz();
	hidden_layer.cala();
	
	auto h_base = h_train_0 + s*output_nodes;		//always point to the start of new sample's hypothesis
 	//calculate delta_3 (output_nodes*1)
 	for (auto l = 0;l<output_nodes;l++)
 	{
 		delta_3[l] = *(h_base+l);			//initialize delta_3 by current hypothesis
 	}
	delta_3[y_train[s]-1] = delta_3[y_train[s]-1] -1;
		
	delta_2.resize(0);					//reinitiallizing delta_2 to zero
	delta_2.resize(hidden_nodes,0);		//delta_2 to zero
	
	/////calculation of delta_2//////
	for(int i = 0; i<output_nodes; i++)
	{
		auto theta_2_base = theta_2_0 + 1 + i*(hidden_nodes+1);		// theta_2_base will point to the 1 next element of theta_2 
																//of every row that is theta_2(1,1), theta_2(2,1) and so on
		for(int j = 0; j<hidden_nodes; j++)
		{
			delta_2[j] += (*(theta_2_base+j)) * delta_3[i]*hidden_layer.sigmoidgrad(j);
		}
	}
	
	 	/*********************************************/
// 	Theta2_grad size = sizeof(theta2) = output_nodes*hidden_nodes
// 	//delta_3 = output_nodes*1 ; a_2 = hidden_nodes*1 ; delta_3* a_2' = output_nodes*hidden_nodes
// 	//theta_grad= theta_grad+ delta_3* a_2'*/
// 	//calculate outer product delta_3* a_2'
 	for(auto j = 0; j<output_nodes; j++){
 		for(auto k =0; k<=hidden_nodes; k++){
 			hidden_layer.theta_grad_temp[(hidden_nodes+1)*j+k] += delta_3[j]*hidden_layer.a[k];
 		}
 	}
	
	for(auto j = 0; j<hidden_nodes; j++){
 		for(auto k =0; k<=input_nodes; k++){
 			input_layer.theta_grad_temp[(input_nodes+1)*j+k] += delta_2[j]*input_layer.a[k];
 		}
 	}
	
 }
 (hidden_layer.theta_grad).resize(0);
 (hidden_layer.theta_grad).resize(hidden_layer.resize_param,0);	
 double * hidden_layer_thetegradtemp_0 = &(hidden_layer.theta_grad_temp[0]);
 double * hidden_layer_thetegrad_0 =&(hidden_layer.theta_grad[0]);
 (input_layer.theta_grad).resize(0);
 (input_layer.theta_grad).resize(input_layer.resize_param,0);
 double * input_layer_thetegradtemp_0 = &(input_layer.theta_grad_temp[0]);
 double * input_layer_thetegrad_0 = &(input_layer.theta_grad[0]);
 MPI_Allreduce(hidden_layer_thetegradtemp_0, hidden_layer_thetegrad_0, hidden_layer.resize_param, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(input_layer_thetegradtemp_0, input_layer_thetegrad_0, input_layer.resize_param, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);

double mult = 0.0;
//double divid = 1.0/num_samples;			//use this for non MPI code
double divid = 1.0/total_samples_train;			//use this for MPI code
for(auto j = 0; j<output_nodes; j++)
{
 	for(auto k =0; k<=hidden_nodes; k++)
	{
		if(k==0)
			mult = 0.0;
		else
			mult =1.0;
		hidden_layer.theta_grad[(hidden_nodes+1)*j+k] = divid*(hidden_layer.theta_grad[(hidden_nodes+1)*j+k] + 
														mult*lambda*hidden_layer.theta[(hidden_nodes+1)*j+k]);
	
	}
}

for(auto j = 0; j<hidden_nodes; j++)
{
 	for(auto k =0; k<=input_nodes; k++)
	{
		if(k==0)
			mult = 0;
		else
			mult =1;
		input_layer.theta_grad[(input_nodes+1)*j+k] = divid*(input_layer.theta_grad[(input_nodes+1)*j+k] + 
														mult*lambda*input_layer.theta[(input_nodes+1)*j+k]);
	
	}
}
}


void predict()
{
	h_test1.resize(output_nodes);
	//////reading vector X_test and y_test///////////
 ifstream mytestfile;
 mytestfile.open("X_test.txt");
 double test =0;
 if (mytestfile.is_open()) 
 {
 while (1)
	{
    mytestfile >> test;
	X_test1.push_back(test);

	if(mytestfile.eof())
		break;
	}
}
mytestfile.close();

	//cout<<X_test1.size()<<endl;

	
 mytestfile.open("y_test.txt");
 if (mytestfile.is_open()) 
 {
 while (1)
	{
    mytestfile >> test;
	y_test1.push_back(test);

	if(mytestfile.eof())
		break;
	}
}
mytestfile.close();
/////////////////////////////////

test_samples1 = X_test1.size()/input_nodes;

vector<double>::iterator X_test_0 = X_test1.begin();
vector<double>::iterator h_test_0 = h_test1.begin();
vector<double>::iterator y_test_0 = y_test1.begin();

feedforward(X_test_0, h_test_0,test_samples1, 1);
//cout<<"y\ttest_label"<<endl;
double accuracy=0;
for(unsigned int i = 0; i <y_test1.size();i++)
{
	if(y_test1[i]==test_label[i])
		accuracy +=1;
		
	//cout<<y_test1[i]<<"\t"<<test_label[i]<<endl;
}
accuracy = 100*(accuracy/(test_samples1));
cout<<"accuracy of the network is: "<<accuracy<<"%"<<endl;

}

//FUNCTION TO READ X AND y FROM TXT FILES TO CORROSPONDING VECTORS
//HOW TO USE:
//readfile(name of the vector X, name of the vector y, "name of the file for X", "name of the file for y")
void readfile(vector<double> &X, vector<double> &y, string X_file, string y_file)
{
	ifstream myReadFile;
	myReadFile.open(X_file);
	double zu =0;
	if (myReadFile.is_open()) 
	{
		while (1)
		{
			myReadFile >> zu;
			X.push_back(zu);

				if(myReadFile.eof())
					break;
		}
	}
	myReadFile.close();
		
	myReadFile.open(y_file);
	if (myReadFile.is_open()) 
	{
		while (1)
		{
			myReadFile >> zu;
			y.push_back(zu);

			if(myReadFile.eof())
				break;
		}
	}
	myReadFile.close();
}

//FUNCTION TO CALCULATE WORK DIVISION PARAMETERS
//HOW TO USE:
//workdivision(name of the vector y)
//affected variables:
//Q, R, X_sendcounts, y_sendcounts, X_displ, y_displ
/* void workdivision(vector<double> &y, int &samples)
{
	samples = y.size();
	Q = ((int) y.size())/psize;
	R = ((int) y.size())%psize;
	for (auto i =0;i<psize;i++)
	{
		if(i<psize-1)
		{
			X_sendcounts[i] = Q*input_nodes;
			y_sendcounts[i] = Q;
			X_displ[i] = i*Q*input_nodes;
			y_displ[i] = i*Q;
		}
		else
		{
			X_sendcounts[i] = (Q+R)*input_nodes;
			y_sendcounts[i] = Q+R;
			X_displ[i] = i*Q*input_nodes;
			y_displ[i] = i*Q;
		}
	}
} */

//WORKDIVISION FOR LEARNING CURVE
void workdivision(int samples)
{
	Q = (samples)/psize;
	R = (samples)%psize;
	for (auto i =0;i<psize;i++)
	{
		
	
		if (i < R)
		{
			
			X_sendcounts[i] = (Q+1)*input_nodes;
			y_sendcounts[i] = Q+1;
			X_displ[i] = i*(Q+1)*input_nodes;
			y_displ[i] = i*(Q+1);		
		}
		else
		{
			X_sendcounts[i] = Q*input_nodes;
			y_sendcounts[i] = Q;
			X_displ[i] = (i*Q+R)*input_nodes;
			y_displ[i] = i*Q+R;
			if(y_sendcounts[i]==0)
			{
				y_displ[i] = 0;
				X_displ[i] = 0;
			}
		}
	}
}

//FUNCTION TO RESIZE AND ALLOCATE VECTOR IN ALL PROCESS
//HOW TO USE:
//vectorallocate(name of the complete vector X in root, name of the complete vector y in root, 
//													name of the vector X to resize and allocate, name of the vector y to resize and allocate)

void vectorallocate(vector<double> &X_root, vector<double> &y_root, vector<double> &X, vector<double> &y )
{
		
	if(myrank<psize-1)
	{
		X.resize(Q*input_nodes,0);
		y.resize(Q,0);
	}
	else
	{
		X.resize((Q+R)*input_nodes,0);
		y.resize(Q+R,0);
	}
	
	MPI_Scatterv(&y_root[0], y_sendcounts, y_displ, MPI_DOUBLE,&y[0], Q+R, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(&X_root[0], X_sendcounts, X_displ, MPI_DOUBLE,&X[0], (Q+R)*input_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

//VECTOR ALLOCATION FOR LEARNING CURVE
void vectorallocate(vector<double> &X_root, vector<double> &y_root, vector<double> &X, vector<double> &y ,int za)
{
		
	/* if (myrank<R)
	{
		X.resize((Q+1)*input_nodes,0);
		y.resize(Q+1,0);
	}
	else
	{
		X.resize(Q*input_nodes,0);
		y.resize(Q,0);
	} */
	X.resize(0);
	y.resize(0);
	X.resize(X_sendcounts[myrank],0);
	y.resize(y_sendcounts[myrank],0);
	if(y.size()==0)
		y[0]=0;
	if(X.size()==0)
		X[0]=0;
	//cout<<myrank<<" should terminate "<<y[0]<<" ysize is "<<y.size()<<endl;
	
	MPI_Scatterv(&y_root[0], y_sendcounts, y_displ, MPI_DOUBLE,&y[0], y_sendcounts[myrank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(&X_root[0], X_sendcounts, X_displ, MPI_DOUBLE,&X[0], X_sendcounts[myrank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void network_init()
{
	input_layer.system_init(&hidden_layer,1);		// pass 1 for outputlayer
	hidden_layer.system_init(&input_layer, &output_layer);
	output_layer.system_init(&hidden_layer,0);		//pass 0 for inputlayer
	
}

void randomvec()
{
	input_layer.randomtheta();
	hidden_layer.randomtheta();
}
////////////////////////////////////////////////////////////////////////////
/**************************PREVIOUSLY DEFINED FUNCTIONS********************/
////////////////////////////////////////////////////////////////////////////
/* void readfile()
{
	ifstream myReadFile;
 myReadFile.open("X_train.txt");
 double zu =0;
 if (myReadFile.is_open()) 
 {
 while (1)
	{
    myReadFile >> zu;
	X_train_ip.push_back(zu);

	if(myReadFile.eof())
		break;
	}
}
myReadFile.close();

	//cout<<X_ip.size()<<endl;

	
 myReadFile.open("y_train.txt");
 if (myReadFile.is_open()) 
 {
 while (1)
	{
    myReadFile >> zu;
	y_train_ip.push_back(zu);

	if(myReadFile.eof())
		break;
	}
}
myReadFile.close();

	//cout<<y_ip.size()<<endl;
} */
