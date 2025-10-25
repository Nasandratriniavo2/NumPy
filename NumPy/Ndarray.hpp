#ifndef __NDARRAY_HPP__
#define __NDARRAY_HPP__

#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

template<typename T>
class Ndarray {
private:
    vector<T> _data;
    vector<size_t> _shape;
    vector<size_t> _strides;
    size_t _ndim;

    void computeStrides();

public:
    Ndarray(const vector<T>& input_data, const vector<size_t>& input_shape);
    
    // Surcharge d'opérateur
    T operator()(initializer_list<size_t> indices) const;    
    Ndarray<T> operator*(const Ndarray<T>& other) const;
    

    void shape_str() const;
    size_t ndim() const;
    vector<size_t> shape() const;
    void reshape(vector<size_t> newShape);

    // Méthodes statiques pour créer des tableaux
    static Ndarray<T> zeros(const vector<size_t>& shape);
    static Ndarray<T> ones(const vector<size_t>& shape);
    static Ndarray<T> arange(T start, T stop, T step = 1);
    static Ndarray<T> eye(size_t n);
    static Ndarray<T> random(const vector<size_t>& shape);
    static Ndarray<T> linspace(T start, T stop, size_t num);

    // Méthodes pour calculer des statistiques
    T sum(void) const;
    T prod() const;
    double mean(void) const;
    T min(void) const;
    T max(void) const;
    Ndarray<double> sqrt() const;

    // Statistiques avancées
    double var() const;
    double std() const;
    T median() const;
    T percentile(double p) const;

    // Opérations élément par élément
    Ndarray<double> exp() const;
    Ndarray<double> log() const;
    Ndarray<T> abs() const;
    Ndarray<double> pow(double exponent) const;

    // Manipulation
    Ndarray<T> flatten() const;

    // Méthodes utilitaires
    Ndarray<T> clip(T min_val, T max_val) const;
    size_t argmin() const;
    size_t argmax() const;
    bool all() const;
    bool any() const;
    template<typename Predicate>
    Ndarray<T> where(Predicate pred, T true_val, T false_val) const;

    // Méthode pour convertir le type
    template<typename U>
    Ndarray<U> astype() const;
};

#include "Ndarray.tpp"

#endif