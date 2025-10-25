#ifndef __NDARRAY_TPP__
#define __NDARRAY_TPP__

#include <algorithm>
#include <cmath>
#include <random>
#include "Ndarray.hpp"

// === Constructeur et gestion interne ===

template<typename T>
Ndarray<T>::Ndarray(const vector<T>& input_data, const vector<size_t>& input_shape) {
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

template<typename T>
void Ndarray<T>::computeStrides() {
    _strides.resize(_ndim);
    if (_ndim == 0) return;
    _strides[_ndim - 1] = 1;

    for (int i = static_cast<int>(_ndim) - 2; i >= 0; --i) {
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
}

// === Accès et opérateurs ===

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
Ndarray<T> Ndarray<T>::operator*(const Ndarray<T>& other) const {
    if (_shape != other._shape) 
        throw invalid_argument("Les formes des tableaux ne correspondent pas pour la multiplication élément par élément.");
    
    vector<T> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) 
        new_data[i] = _data[i] * other._data[i];
    
    return Ndarray<T>(new_data, _shape);
}

// === Forme ===

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

// === Création statique ===

template<typename T>
Ndarray<T> Ndarray<T>::zeros(const vector<size_t>& shape) {
    if (shape.empty()) {
        throw invalid_argument("La forme ne peut pas être vide.");
    }

    size_t total_size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("Les dimensions doivent être non nulles.");
        }
        total_size *= dim;
    }

    vector<T> data(total_size, 0);
    return Ndarray<T>(data, const_cast<vector<size_t>&>(shape));
}

template<typename T>
Ndarray<T> Ndarray<T>::ones(const vector<size_t>& shape) {
    if (shape.empty()) {
        throw invalid_argument("La forme ne peut pas être vide.");
    }

    size_t total_size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("Les dimensions doivent être non nulles.");
        }
        total_size *= dim;
    }

    vector<T> data(total_size, 1);
    return Ndarray<T>(data, const_cast<vector<size_t>&>(shape));
}

template<typename T>
Ndarray<T> Ndarray<T>::arange(T start, T stop, T step) {
    if (step == 0) {
        throw invalid_argument("Le pas ne peut pas être nul.");
    }

    size_t size = static_cast<size_t>(std::max(0.0, ceil(static_cast<double>(stop - start) / step)));
    vector<T> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = start + i * step;
    }

    vector<size_t> shape = {size};
    return Ndarray<T>(data, shape);
}

template<typename T>
Ndarray<T> Ndarray<T>::eye(size_t n) {
    if (n == 0) {
        throw invalid_argument("La taille de la matrice identité doit être positive.");
    }
    vector<T> data(n * n, 0);
    for (size_t i = 0; i < n; ++i) {
        data[i * n + i] = 1;
    }
    vector<size_t> shape = {n, n};
    return Ndarray<T>(data, shape);
}

template<typename T>
Ndarray<T> Ndarray<T>::random(const vector<size_t>& shape) {
    if (shape.empty()) {
        throw invalid_argument("La forme ne peut pas être vide.");
    }

    size_t total_size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("Les dimensions doivent être non nulles.");
        }
        total_size *= dim;
    }

    vector<T> data(total_size);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

    for (size_t i = 0; i < total_size; ++i) {
        data[i] = static_cast<T>(dis(gen));
    }

    return Ndarray<T>(data, const_cast<vector<size_t>&>(shape));
}

template<typename T>
Ndarray<T> Ndarray<T>::linspace(T start, T stop, size_t num) {
    if (num == 0) {
        throw invalid_argument("Le nombre d'échantillons doit être positif.");
    }
    if (num == 1) {
        vector<T> data(1, start);
        return Ndarray<T>(data, vector<size_t>{1});
    }

    vector<T> data(num);
    double step = static_cast<double>(stop - start) / static_cast<double>(num - 1);
    for (size_t i = 0; i < num; ++i) {
        data[i] = start + static_cast<T>(step * i);
    }

    return Ndarray<T>(data, vector<size_t>{num});
}

// === Conversion de type ===

template<typename T>
template<typename U>
Ndarray<U> Ndarray<T>::astype() const {
    vector<U> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        new_data[i] = static_cast<U>(_data[i]);
    }
    return Ndarray<U>(new_data, _shape);
}

// === Statistiques de base ===

template<typename T>
T Ndarray<T>::sum(void) const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la somme d'un tableau vide.");
    T result = 0;
    for (const T& val : _data) 
        result += val;
    return result;
}

template<typename T>
T Ndarray<T>::prod(void) const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer le produit d'un tableau vide.");
    T result = 1;
    for (const T& val : _data) 
        result *= val;
    return result;
}

template<typename T>
double Ndarray<T>::mean() const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la moyenne d'un tableau vide.");
    return static_cast<double>(sum()) / _data.size();
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
Ndarray<double> Ndarray<T>::sqrt() const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la racine carrée d'un tableau vide.");
    vector<double> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        if (_data[i] < 0) {
            throw runtime_error("Racine carrée d'une valeur négative non définie.");
        }
        new_data[i] = std::sqrt(static_cast<double>(_data[i]));
    }
    return Ndarray<double>(new_data, _shape);
}

// === Statistiques avancées ===

template<typename T>
double Ndarray<T>::var() const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la variance d'un tableau vide.");
    double m = mean();
    double sum_sq = 0.0;
    for (const T& val : _data) {
        double diff = static_cast<double>(val) - m;
        sum_sq += diff * diff;
    }
    return sum_sq / _data.size();
}

template<typename T>
double Ndarray<T>::std() const {
    return std::sqrt(var());
}

template<typename T>
T Ndarray<T>::median() const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer la médiane d'un tableau vide.");
    vector<T> sorted = _data;
    sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / static_cast<T>(2);
    } else {
        return sorted[n/2];
    }
}

template<typename T>
T Ndarray<T>::percentile(double p) const {
    if (_data.empty()) 
        throw runtime_error("Impossible de calculer le percentile d'un tableau vide.");
    if (p < 0.0 || p > 100.0) 
        throw invalid_argument("Le percentile doit être entre 0 et 100.");
    
    vector<T> sorted = _data;
    sort(sorted.begin(), sorted.end());
    double index = p * (static_cast<double>(sorted.size()) - 1) / 100.0;
    size_t i = static_cast<size_t>(index);
    double frac = index - i;

    if (i + 1 >= sorted.size()) {
        return sorted[i];
    } else {
        return static_cast<T>(
            static_cast<double>(sorted[i]) * (1.0 - frac) +
            static_cast<double>(sorted[i + 1]) * frac
        );
    }
}

// === Opérations élément par élément ===

template<typename T>
Ndarray<double> Ndarray<T>::exp() const {
    if (_data.empty()) 
        throw runtime_error("Tableau vide dans exp().");
    vector<double> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        new_data[i] = std::exp(static_cast<double>(_data[i]));
    }
    return Ndarray<double>(new_data, _shape);
}

template<typename T>
Ndarray<double> Ndarray<T>::log() const {
    if (_data.empty()) 
        throw runtime_error("Tableau vide dans log().");
    vector<double> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        if (_data[i] <= 0) {
            throw runtime_error("Logarithme d'une valeur non positive.");
        }
        new_data[i] = std::log(static_cast<double>(_data[i]));
    }
    return Ndarray<double>(new_data, _shape);
}

template<typename T>
Ndarray<T> Ndarray<T>::abs() const {
    if (_data.empty()) 
        throw runtime_error("Tableau vide dans abs().");
    vector<T> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        new_data[i] = std::abs(_data[i]);
    }
    return Ndarray<T>(new_data, _shape);
}

template<typename T>
Ndarray<double> Ndarray<T>::pow(double exponent) const {
    if (_data.empty()) 
        throw runtime_error("Tableau vide dans pow().");
    vector<double> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        new_data[i] = std::pow(static_cast<double>(_data[i]), exponent);
    }
    return Ndarray<double>(new_data, _shape);
}

// === Manipulation ===

template<typename T>
Ndarray<T> Ndarray<T>::flatten() const {
    vector<T> flat_data = _data;
    vector<size_t> new_shape = {_data.size()};
    return Ndarray<T>(flat_data, new_shape);
}

// === Méthodes utilitaires ===

template<typename T>
Ndarray<T> Ndarray<T>::clip(T min_val, T max_val) const {
    if (min_val > max_val) {
        throw invalid_argument("min_val ne peut pas être supérieur à max_val.");
    }
    vector<T> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        if (_data[i] < min_val) new_data[i] = min_val;
        else if (_data[i] > max_val) new_data[i] = max_val;
        else new_data[i] = _data[i];
    }
    return Ndarray<T>(new_data, _shape);
}

template<typename T>
size_t Ndarray<T>::argmin() const {
    if (_data.empty()) 
        throw runtime_error("Tableau vide dans argmin().");
    size_t idx = 0;
    T min_val = _data[0];
    for (size_t i = 1; i < _data.size(); ++i) {
        if (_data[i] < min_val) {
            min_val = _data[i];
            idx = i;
        }
    }
    return idx;
}

template<typename T>
size_t Ndarray<T>::argmax() const {
    if (_data.empty()) 
        throw runtime_error("Tableau vide dans argmax().");
    size_t idx = 0;
    T max_val = _data[0];
    for (size_t i = 1; i < _data.size(); ++i) {
        if (_data[i] > max_val) {
            max_val = _data[i];
            idx = i;
        }
    }
    return idx;
}

template<typename T>
bool Ndarray<T>::all() const {
    for (const T& val : _data) {
        if (!val) return false;
    }
    return true;
}

template<typename T>
bool Ndarray<T>::any() const {
    for (const T& val : _data) {
        if (val) return true;
    }
    return false;
}

template<typename T>
template<typename Predicate>
Ndarray<T> Ndarray<T>::where(Predicate pred, T true_val, T false_val) const {
    vector<T> new_data(_data.size());
    for (size_t i = 0; i < _data.size(); ++i) {
        new_data[i] = pred(_data[i]) ? true_val : false_val;
    }
    return Ndarray<T>(new_data, _shape);
}

#endif