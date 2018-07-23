#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include <iostream>
#include <vector>
#include <math.h>
//#include <mpi.h>
#include <fstream>

using namespace std;
//define double real;
class neural_net
{
	public:

	int numnodes_current;
	int numnodes_next;
	int resize_param;
	double sigradz;
	vector<double> a;
	vector<double> z;
	vector<double> theta;
	vector<double> theta_old;			//used to store the theta of previous iteration in backpropagation
	vector<double> theta_grad;			//will have theta grad after MPI_allreduce
	vector<double> theta_grad_temp;		//to store the thetagrad of individual MPI process
	vector<double> delta_theta;
	neural_net* previous_layer;
	neural_net* next_layer;
	neural_net(int,int);
	neural_net(int);
	double sigmoid(double);
	void randomtheta();
	void calz();
	void cala();
	void cala(vector<double>::iterator);
	int calhypothesis(vector<double>::iterator);
	double costfun(vector<double>::iterator, vector<double>::iterator,int);
	void system_init(neural_net*, neural_net*);
	void system_init(neural_net*, int);
	double sigmoidgrad(int);
	void updatetheta(double);
	void theta_eq(vector<double> &p, vector<double> &q);		//a = b

	//double get(int i,int j);

};

#endif
