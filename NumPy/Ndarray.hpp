#ifndef __NDARRAY_HPP__
#define __NDARRAY_HPP__

#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <variant>

using namespace std;

/* // Forward declaration nécessaire pour utiliser NestedList
template<typename T>
class Ndarray;
 */
// Définition du type imbriqué récursif avant la classe
/* template<typename T>
using NestedList = variant<T, vector<NestedList<T>>>;
 */
template<typename T>
class Ndarray {
/* public:
    // Utilise le type déjà défini en amont
    using NestedListType = variant<T, vector<NestedList<T>>>;
 */
private:
    vector<T> _data;
    vector<size_t> _shape;
    vector<size_t> _strides;
    size_t _ndim;

    void computeStrides();
    /* void flattenAndSetShape(const vector<NestedListType>& list, vector<T>& data, vector<size_t>& shape, size_t depth);
    void validateShape(const vector<NestedListType>& list, const vector<size_t>& shape, size_t depth);
 */
public:
    Ndarray(vector<T>& input_data, vector<size_t>& input_shape);
    //Ndarray(initializer_list<NestedListType> init_list);
    
    //Surcharge d'operateur
    T operator()(initializer_list<size_t> indices) const;    
    Ndarray<T> operator*(const Ndarray<T>& other) const;
    

    void shape_str() const;
    size_t ndim() const;
    vector<size_t> shape() const;
    void reshape(vector<size_t> newShape);

    //Methodes statiques pour creer des tableaux
    static Ndarray<T> zeros(const vector<size_t>& shape);
    static Ndarray<T> ones(const vector<size_t>& shape);
    static Ndarray<T> arange(T start, T stop, T step = 1);

    //Methodes pour calculer des statistiques
    T sum(void) const; //Calcul la somme de tous les elements du tableau
    T prod() const; //Calcul la somme de tous les elements du tableau
    double mean(void) const; //Calcul la moyenne de tous les elements du tableau
    T min(void) const; //Retourne le minmum
    T max(void) const; //Retourne le maximum
    Ndarray<double> sqrt() const;

    // Methode pour convertir le type
    template<typename U>
    Ndarray<U> astype() const;

};

#include "Ndarray.tpp"

#endif 