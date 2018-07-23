#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <time.h>
#include <mpi.h>
#include "neural_net.h"
using namespace std;


neural_net::neural_net(int number_of_currentlayer_nodes, int number_of_nextlayer_nodes)
{
	numnodes_current = number_of_currentlayer_nodes;
	numnodes_next = number_of_nextlayer_nodes;
	resize_param = numnodes_next*(numnodes_current+1);
	//theta.resize(numnodes_next*(numnodes_current+1),0);
	theta.resize(resize_param,0);
	theta_old.resize(resize_param,0);
	//theta_grad.resize(numnodes_next*(numnodes_current+1),0);
	theta_grad.resize(resize_param,0);
	//theta_grad_temp.resize(numnodes_next*(numnodes_current+1),0);
	theta_grad_temp.resize(resize_param,0);
	delta_theta.resize(numnodes_current,0);
	z.resize(numnodes_current,0);	//layer input
	a.resize(numnodes_current+1,0);	//layer output
};

neural_net::neural_net(int output_layer_nodes)
{
	numnodes_current = output_layer_nodes;
	z.resize(numnodes_current,0);	//layer input
	a.resize(numnodes_current,0);	//layer output
};

void neural_net::system_init(neural_net*p , neural_net*n )
{
	previous_layer = p;
	next_layer = n;
};
void neural_net::system_init(neural_net* layer,int x)		//x will decide which layer to assign
{
	if(x==0)
		previous_layer = layer;
	else
		next_layer = layer;
};

double neural_net::sigmoid(double x)
{
	double temp = exp(-x);
	double val = 1.0/(1.0+temp);
	return val;
};

void neural_net::randomtheta()
{
	double num = 0;
	double eps = sqrt(6.0/((double) numnodes_current +(double) numnodes_next));
	for (unsigned int i = 1;i<=theta.size();i++)
	{
		srand(i);
		num = 2*fmod(rand(),eps)-eps;		// "%" is only defined for integer operators, use fmod(double,double) for floting point operators
		theta[i-1] = num;
	}

 /* cout<<"THETA"<<endl;
 	for (auto it = theta.begin();it!=theta.end();it++)
 		cout<<*it<<endl; */
};

void neural_net::calz()
{
	vector<double>::iterator abegin = previous_layer->a.begin();
	vector<double>::iterator thetabegin = previous_layer->theta.begin();
	int nodestemp = previous_layer->numnodes_current;
	int nodes = nodestemp+1; //numnodes_current +1 to include bias node
	for(unsigned int i=0;i<z.size();i++)
	{z[i] = 0;
		for(int j=0;j<nodes;j++)
		{
			z[i] += (*(abegin+j)) * (*(thetabegin+j+nodes*i));
		}
	}
}

void neural_net::cala()
{
	a[0] = 1.0;

	for (unsigned int i =1;i<a.size();i++)
	{
		a[i] = sigmoid(z[i-1]);
	}
};

void neural_net::cala(vector<double>::iterator abegin)
{
	a[0] = 1.0;

	for (unsigned int i =1;i<a.size();i++)
	{
		a[i] = *(abegin+i-1);
	}
};

int neural_net::calhypothesis(vector<double>::iterator hypo)
{
	double max = 0.0;
	int index = 0;
	for (int i =0;i<numnodes_current;i++)
	{
		//*(hypo+i) = sigmoid(z[i]);
		*(hypo+i) = sigmoid(z[i]);
		if(max< *(hypo+i))
		{
			max = *(hypo+i);
			index = i;
		}		
	}
	return (index+1);
};


double neural_net::costfun(vector<double>::iterator h, vector<double>::iterator y,int num_samples)
{
	double J = 0;	//network cost
	for(auto s=0;s<num_samples;s++)
	{ 	auto label = *(y+s);	//value of Y label
		auto h_0 = h+s*numnodes_current;	//h_0 point to the start of a new sample
		for(auto l=0;l<numnodes_current;l++)
		{
			if(l == label-1)
			{
				J = J - log(*(h_0 + l));
			}
			if(l != label-1)
			{
				J = J - log(1.0-(*(h_0 + l)));
			}
		}
	}
	return (J);
};

double neural_net::sigmoidgrad(int i)
{
	return (a[i+1])*(1-a[i+1]);
};

void neural_net::updatetheta(double step)
{
	auto theta_grad_iter = theta_grad.begin();
	auto theta_iter = theta.begin();
	while(theta_iter<theta.end())
	{
		*theta_iter -= step*(*theta_grad_iter);
		theta_grad_iter++;
		theta_iter++;
	}
};

void neural_net::theta_eq(vector<double> &p, vector<double> &q)		//a=b
{
	auto it_new = p.begin(); 
	for (auto it = q.begin();it!=q.end();it++)
	{
		*it_new = *it;
	}
};
