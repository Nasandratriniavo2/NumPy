#ifndef __NDARRAY_TPP__
#define __NDARRAY_TPP__
#include <algorithm>
#include <cmath>
#include "Ndarray.hpp"

template<typename T>
template<typename U>
Ndarray<U> Ndarray<T>::astype() const {
    // Créer un nouveau vecteur de données avec le type cible
    vector<U> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        new_data[i] = static_cast<U>(_data[i]);
    }

    // Copier la forme
    vector<size_t> new_shape = _shape;

    // Retourner un nouvel Ndarray avec le type U
    return Ndarray<U>(new_data, new_shape);
}

template<typename T>
Ndarray<double> Ndarray<T>::sqrt() const {
    vector<double> new_data(_data.size());

    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la racine carrée d'un tableau vide.");

    //Créer un nouveau vecteur de données de type double
    for (size_t i = 0; i < _data.size(); ++i) {
        if (_data[i] < 0) {
            throw runtime_error("Racine carrée d'une valeur négative non définie.");
        }
        new_data[i] = std::sqrt(static_cast<double>(_data[i]));
    }

    // Copier la forme
    vector<size_t> new_shape = _shape;

    // Retourner un nouvel Ndarray avec le type double
    return Ndarray<double>(new_data, new_shape);
}

template<typename T>
T Ndarray<T>::max(void) const {
    if (_data.empty()) {
        throw runtime_error("Impossible de calculer le maximum d'un tableau vide.");
    }

    T result = _data[0];
    for (size_t i = 1; i < _data.size(); ++i) {
        if (_data[i] > result) {
            result = _data[i];
        }
    }
    return result;
}

template<typename T>
T Ndarray<T>::min(void) const {
    if (_data.empty()) {
        throw runtime_error("Impossible de calculer le minimum d'un tableau vide.");
    }

    T result = _data[0];
    for (size_t i = 1; i < _data.size(); ++i) {
        if (_data[i] < result) {
            result = _data[i];
        }
    }
    return result;
}

template<typename T>
double Ndarray<T>::mean() const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la moyenne d'un tableau vide.");
    
    return static_cast<double>(sum()) / _data.size();
}

template<typename T>
T Ndarray<T>::prod(void) const {
    T result = 1;

    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la somme d'un tableau vide.");
    for (const T& val : _data) 
        result *= val;
    
    return result;
}

template<typename T>
T Ndarray<T>::sum(void) const {
    T result = 0;

    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la somme d'un tableau vide.");
    for (const T& val : _data) 
        result += val;
    
    return result;
}

template<typename T>
Ndarray<T> Ndarray<T>::arange(T start, T stop, T step) {
    if (step == 0) {
        throw invalid_argument("Le pas ne peut pas être nul.");
    }

    // Calculer la taille du tableau
    size_t size = static_cast<size_t>(std::max(0.0, ceil(static_cast<double>(stop - start) / step)));

    // Créer le vecteur de données
    vector<T> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = start + i * step;
    }

    // Définir la forme (1D)
    vector<size_t> shape = {size};
    
    return Ndarray<T>(data, shape);
}


template<typename T>
Ndarray<T> Ndarray<T>::ones(const vector<size_t>& shape) {
    // Vérifier que la forme n'est pas vide
    if (shape.empty()) {
        throw invalid_argument("La forme ne peut pas être vide.");
    }

    // Calculer la taille totale
    size_t total_size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("Les dimensions doivent être non nulles.");
        }
        total_size *= dim;
    }

    vector<T> data(total_size, 1);

    // Retourner un nouvel Ndarray
    return Ndarray<T>(data, const_cast<vector<size_t>&>(shape));
}

template<typename T>
Ndarray<T> Ndarray<T>::zeros(const vector<size_t>& shape) {
    // Vérifier que la forme n'est pas vide
    if (shape.empty()) {
        throw invalid_argument("La forme ne peut pas être vide.");
    }

    // Calculer la taille totale
    size_t total_size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("Les dimensions doivent être non nulles.");
        }
        total_size *= dim;
    }

    // Créer un vecteur de zéros
    vector<T> data(total_size, 0);

    return Ndarray<T>(data, const_cast<vector<size_t>&>(shape));
}

template<typename T>
Ndarray<T>::Ndarray(vector<T>& input_data, vector<size_t>& input_shape) {
    _data = input_data;
    _shape = input_shape;
    _ndim = _shape.size();

    size_t total_size = 1;
    for (size_t dim : _shape)
        total_size *= dim;

    if (total_size != _data.size())
        throw invalid_argument("La taille des données ne correspond pas au shape fourni.");

    computeStrides();
}

/* template<typename T>
Ndarray<T>::Ndarray(initializer_list<NestedListType> init_list) {
    vector<NestedListType> nested_list(init_list.begin(), init_list.end());
    vector<T> data;
    vector<size_t> shape;

    flattenAndSetShape(nested_list, data, shape, 0);
    validateShape(nested_list, shape, 0);

    _data = data;
    _shape = shape;
    _ndim = shape.size();
    computeStrides();
}
 */
template<typename T>
void Ndarray<T>::computeStrides() {
    _strides.resize(_ndim);
    _strides[_ndim - 1] = 1;

    for (int i = _ndim - 2; i >= 0; --i) {
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
}

/* template<typename T>
void Ndarray<T>::flattenAndSetShape(const vector<NestedListType>& list, vector<T>& data, vector<size_t>& shape, size_t depth) {
    if (list.empty()) return;

    if (depth >= shape.size()) {
        shape.push_back(list.size());
    } else if (shape[depth] != list.size()) {
        throw invalid_argument("Dimensions incohérentes dans la liste imbriquée.");
    }

    for (const auto& item : list) {
        if (holds_alternative<T>(item)) {
            data.push_back(get<T>(item));
        } else {
            const auto& sublist = get<vector<NestedListType>>(item);
            flattenAndSetShape(sublist, data, shape, depth + 1);
        }
    }
}

template<typename T>
void Ndarray<T>::validateShape(const vector<NestedListType>& list, const vector<size_t>& shape, size_t depth) {
    if (depth >= shape.size() || list.size() != shape[depth]) {
        throw invalid_argument("Taille de liste incohérente avec la forme calculée.");
    }

    for (const auto& item : list) {
        if (!holds_alternative<vector<NestedListType>>(item)) continue;

        const auto& sublist = get<vector<NestedListType>>(item);
        if (depth + 1 < shape.size()) {
            validateShape(sublist, shape, depth + 1);
        }
    }
}
 */
template<typename T>
T Ndarray<T>::operator()(initializer_list<size_t> indices) const {
    if (indices.size() != _ndim) {
        throw out_of_range("Nombre d'indices incorrect.");
    }

    size_t flat_index = 0;
    size_t i = 0;
    for (size_t idx : indices) {
        if (idx >= _shape[i]) {
            throw out_of_range("Indice hors limites.");
        }
        flat_index += idx * _strides[i++];
    }

    return _data[flat_index];
}

template<typename T>
void Ndarray<T>::shape_str() const {
    cout << "(";
    for (size_t i = 0; i < _ndim; ++i) {
        cout << _shape[i];
        if (i < _ndim - 1) cout << ", ";
    }
    cout << ")";
}

template<typename T>
size_t Ndarray<T>::ndim() const {
    return _ndim;
}

template<typename T>
vector<size_t> Ndarray<T>::shape() const {
    return _shape;
}

template<typename T>
void Ndarray<T>::reshape(vector<size_t> newShape) {
    size_t new_size = 1;
    for (size_t dim : newShape) new_size *= dim;

    if (new_size != _data.size()) {
        throw invalid_argument("Le nouveau shape ne correspond pas à la taille des données.");
    }

    _shape = newShape;
    _ndim = newShape.size();
    computeStrides();
}

template<typename T>
Ndarray<T> Ndarray<T>::operator*(const Ndarray<T>& other) const {
    vector<T> new_data(_data.size());
    vector<size_t> new_shape = _shape;

    if (_shape != other._shape) 
        throw invalid_argument("Les formes des tableaux ne correspondent pas pour la multiplication élément par élément.");
    // Créer un nouveau vecteur de données
    for (size_t i = 0; i < _data.size(); ++i) 
        new_data[i] = _data[i] * other._data[i];
    
    return Ndarray<T>(new_data, new_shape);
}


#endif 