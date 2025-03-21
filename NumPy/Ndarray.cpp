#include "Ndarray.hpp"

template<typename T>
T Ndarray<T>::at(const std::vector<size_t>& indices) {
    size_t flat_index = 0;
    size_t i;
    if (indices.size() != ndim) 
        throw std::out_of_range("Nombre d'indices incorrect.");
    //Pour avoir l'indice equilvalent dans le tableau applatie
    for (i = 0; i < ndim; ++i) 
        flat_index += indices[i] * strides[i];
    
    return data[flat_index];
}

template<typename T>
void Ndarray<T>::printShape(void) {
    size_t i;
    cout << "(";
    for (i = 0; i < ndim; ++i) {
        cout << shape[i];
        if (i < ndim - 1) 
            cout << ", ";
    }
    cout << ")" << endl;
}

template<typename T>
size_t Ndarray<T>::getNdim(void) {
    return ndim;
}

template<typename T>
vector<size_t> Ndarray<T>::getShape(void) {
    return shape;
}

template<typename T>
void Ndarray<T>::reshape(vector<T> newShape) {
    // Vérification de la cohérence entre data et newshape
    size_t new_size = 1;
    for (size_t dim : newShape)    
        new_size *= dim;
        
    if (new_size != data.size()) 
        throw std::invalid_argument("Le nouveau shape ne correspond pas à la taille des données.");
        
    shape = newShape;
    ndim = shape.size();
    computeStrides();
}

template<typename T>
Ndarray<T>::Ndarray(vector<T>& input_data, vector<size_t>& input_shape) {
    data = input_data;
    shape = input_shape;
    ndim = shape.size();

    // Vérification de la cohérence entre data et shape
    size_t total_size = 1;
    for (size_t dim : shape) 
        total_size *= dim;
    
    if (total_size != data.size()) 
        throw std::invalid_argument("La taille des données ne correspond pas au shape fourni.");
    
    computeStrides();
}

template<typename T>
void Ndarray<T>::computeStrides(void) {
    int i;
    strides.resize(ndim); //Meme dimension que le shape
    strides[ndim - 1] = 1; //Toujours 1

    for (i = ndim - 2; i >= 0; i--) 
        strides[i] = strides[i + 1] * shape [i + 1];
}



