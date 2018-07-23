#include<iostream>
#include<math.h>
#include <vector>
#include "neural_net.h"
#include "neural_net.cpp"
#include "function.h"
#include <mpi.h>
#include<sys/time.h>
using namespace std;

int main(int argc, char *argv[])
{
	/*for(int i=0;i<1;i++)
	{
		vector<double> vec;
		vec.resize(0,0);
		cout<<"start vector value is "<<vec[0]<<endl;
	}*/
	MPI_Init(&argc,&argv);
	MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
	MPI_Comm_size( MPI_COMM_WORLD,&psize);
	X_sendcounts = new int[psize];
	X_displ = new int[psize];
	y_sendcounts = new int[psize];
	y_displ = new int[psize];
	///////////////////////////////////////////////////////////////
	/////////read file and work division for training data/////////
	//////////////////////////////////////////////////////////////
	if (myrank ==0)
	{
		//readfile();
		readfile(X_train_ip, y_train_ip, "X_train.txt", "y_train.txt");
		total_samples_train = y_train_ip.size();
		//cout<<"total train samples are "<<total_samples_train<<endl;
		//workdivision(y_train_ip, total_samples_train);
		workdivision(total_samples_train);
		
	}
	//synchronize the value of Q, R and total_samples
	MPI_Bcast(&total_samples_train,1,MPI_INT,0,MPI_COMM_WORLD);
/* 	MPI_Bcast(&Q,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&R,1,MPI_INT,0,MPI_COMM_WORLD);*/
	
	MPI_Bcast(X_sendcounts,psize,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(y_sendcounts,psize,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(X_displ,psize,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(y_displ,psize,MPI_INT,0,MPI_COMM_WORLD);
	//MPI_Bcast(X_sendcounts,psize,MPI_INT,0,MPI_COMM_WORLD);
	//MPI_Bcast(y_sendcounts,psize,MPI_INT,0,MPI_COMM_WORLD);
	//MPI_Bcast(X_displ,psize,MPI_INT,0,MPI_COMM_WORLD);
	//MPI_Bcast(y_displ,psize,MPI_INT,0,MPI_COMM_WORLD);
		
	vectorallocate(X_train_ip,y_train_ip, X_train,y_train,1);
	num_samples_train = y_train.size();				//number of samples in each process
	h_train.resize(num_samples_train*output_nodes);	//size of hypothesis vector for each process
	if (myrank ==0)
		cout<<"total training samples is: "<<total_samples_train<<endl;
	//cout<<"\nmy rank is: "<<myrank<<" and my y_sendcount for training data is: "<<y_sendcounts[myrank]<<" and it should be equal to "<<y_train.size();
	MPI_Barrier(MPI_COMM_WORLD);
	
	////////////////////////////////////////////////////////////////
	//////////read file and work division for test data/////////////
	////////////////////////////////////////////////////////////////
	if (myrank ==0)
	{
		//readfile();
		readfile(X_test_ip, y_test_ip, "X_test.txt", "y_test.txt");
		total_samples_test = y_test_ip.size();
		workdivision(total_samples_test);
		
	}
	
	//synchronize the value of Q, R and total_samples
	/*MPI_Bcast(&Q,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&R,1,MPI_INT,0,MPI_COMM_WORLD);
	*/
	MPI_Bcast(&total_samples_test,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(X_sendcounts,psize,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(y_sendcounts,psize,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(X_displ,psize,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(y_displ,psize,MPI_INT,0,MPI_COMM_WORLD);
	
	vectorallocate(X_test_ip,y_test_ip, X_test,y_test,1);
	num_samples_test = y_test.size();
	h_test.resize(num_samples_test*output_nodes);
	if (myrank ==0)
		cout<<"\ntotal test samples is: "<<total_samples_test<<endl;
	//cout<<"\nmy rank is: "<<myrank<<" and my y_sendcount for test data is: "<<y_sendcounts[myrank]<<" and it should be equal to "<<y_test.size();
	MPI_Barrier(MPI_COMM_WORLD);
	
	network_init();		//edit this function if you want to add more hidden layers
	randomvec();		//this function randomize the theta vector of all the layers. EDIT THIS FUNCTION TO ADD MORE HIDDEN LAYERS
	/* input_layer.system_init(&hidden_layer,1);		// pass 1 for outputlayer
	hidden_layer.system_init(&input_layer, &output_layer);
	output_layer.system_init(&hidden_layer,0);		//pass 0 for inputlayer
	input_layer.randomtheta();
	hidden_layer.randomtheta(); */
	vector<double>::iterator X_train_0 = X_train.begin();
	vector<double>::iterator h_train_0 = h_train.begin();
	vector<double>::iterator y_train_0 = y_train.begin();
	
	

	vector<double>::iterator X_test_0 = X_test.begin();
	vector<double>::iterator h_test_0 = h_test.begin();
	vector<double>::iterator y_test_0 = y_test.begin();
	
	feedforward(X_train_0, h_train_0, num_samples_train);
	
	//double J_old_temp = 0.0;
	double J_new_train_temp = 0.0;
	double J_test_temp = 0.0;
	
	J_new_train_temp = output_layer.costfun(h_train_0,y_train_0,num_samples_train);
	
	J_new_train = 0;		//in general check that receive buffer is initialized to zero before performing MPI_Allreduce
	MPI_Allreduce(&J_new_train_temp, &J_new_train, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
	J_new_train = J_new_train/((double) total_samples_train);
	if(myrank==0)
	{
		cout<<"cost before the training is: "<<J_new_train<<endl<<endl;
		//cout<<"J_current\t\tJ_old\n"<<endl;
	}
	int iter = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank==0){
		cout<< "Training has started!" << endl;
		wcTimeStart = getSeconds();
	}
		do
		{
			hidden_layer.theta_eq(hidden_layer.theta_old, hidden_layer.theta);		//theta_new = theta
			input_layer.theta_eq(input_layer.theta_old, input_layer.theta);			//theta_new = theta
			J_old_train = J_new_train;
			backpropagation(X_train_0,h_train_0, num_samples_train);
			hidden_layer.updatetheta(t_k);
			input_layer.updatetheta(t_k);
			feedforward(X_train_0,h_train_0, num_samples_train);
			J_new_train_temp = 0;
			J_new_train_temp = output_layer.costfun(h_train_0,y_train_0,num_samples_train);
			J_new_train = 0;		//in general check that receive buffer is initialized to zero before performing MPI_Allreduce
			MPI_Allreduce(&J_new_train_temp, &J_new_train, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
			J_new_train = J_new_train/((double) total_samples_train);
			/*if (myrank==0)
				cout<<J_new_train<<"\t\t"<<J_old_train<<endl;*/

			if((J_new_train-J_old_train)>0)
			{
				hidden_layer.theta_eq(hidden_layer.theta,hidden_layer.theta_old);		//theta = theta_new
				input_layer.theta_eq( input_layer.theta, input_layer.theta_old);
			    cout<<"break: "<<J_old_train<<" "<<J_new_train<<endl;
				break;
			}
			
			iter++;
			
		}
		while(J_new_train>0.1);
	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank==0){
		wcTimeEnd = getSeconds();
		cout<< "\nTraining Complete! " << "\nValue of J " << J_new_train << endl;
		cout<<"current lambda is: "<<lambda<<" and total iteration: "<<iter<<endl;
		cout<< "Time for training " << total_samples_train << " :" << wcTimeEnd-wcTimeStart << " sec"<<endl;
		cout<< "\nTesting data set started: " << endl;
		wcTimeStart = getSeconds();
		predict();			//theta = theta_new
		wcTimeEnd = getSeconds();
		cout <<"\nTesting complete!" << endl;
		cout <<"Time taken for testing " << total_samples_test << " :" << wcTimeEnd-wcTimeStart << " sec"<<endl;
		/* ofstream theta1_write;
		theta1_write.open("theta_1.txt");
		for (auto it = (input_layer.theta).begin(); it != (input_layer.theta).end(); it++)
		{
			theta1_write<<*it<<endl;
		}
		theta1_write.close();
	
		ofstream theta2_write;
		theta2_write.open("theta_2.txt");
		for (auto it = (hidden_layer.theta).begin(); it != (hidden_layer.theta).end(); it++)
		{
			theta2_write<<*it<<endl;
		}
		theta2_write.close();	 */
	}
																				
	MPI_Finalize();
	return 0;
	
}
