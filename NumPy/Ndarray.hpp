#ifndef __NDARRAY__
#define __NDARRAY__
#include<iostream>
#include <vector>
using namespace std;

template<typename T>
class Ndarray {
private:
    vector<T> data; //Tableau applatie
    vector<size_t> shape; //Shape des tableau
    vector<size_t> strides; //Contient les strides pour les dimensions
    size_t ndim; //Nombre de dimension

    //Calcul les strides
    void computeStrides(void);
public:
    Ndarray(vector<T>& input_data, vector<size_t>& input_shape);
    void reshape(vector<T> newShape);
    vector<size_t> getShape(void);
    size_t getNdim(void);
    void printShape(void);
    T at(const std::vector<size_t>& indices);
};

#endif