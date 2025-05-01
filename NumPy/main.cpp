#include "Ndarray.hpp"
#include <iostream>
using namespace std;

int main() {
    // Tester zeros
    Ndarray<int> zeros = Ndarray<int>::zeros({2, 3});
    cout << "Tableau de zéros (int) : ";
    zeros.shape_str(); // Affiche : (2, 3)
    cout << endl << "Somme : " << zeros.sum() << endl; // Affiche : 0
    cout << "Produit : " << zeros.prod() << endl; // Affiche : 0

    // Convertir zeros en double
    Ndarray<double> zeros_double = zeros.astype<double>();
    cout << "Tableau de zéros (double) : ";
    zeros_double.shape_str(); // Affiche : (2, 3)
    cout << endl << "Somme : " << zeros_double.sum() << endl; // Affiche : 0.0
    cout << "Produit : " << zeros_double.prod() << endl; // Affiche : 0.0

    // Tester arange
    Ndarray<int> seq = Ndarray<int>::arange(1, 5, 1);
    cout << "Séquence arange(1, 5, 1) (int) : ";
    seq.shape_str(); // Affiche : (4)
    cout << endl << "Éléments : ";
    for (size_t i = 0; i < seq.shape()[0]; ++i) {
        cout << seq({i}) << " "; // Affiche : 1 2 3 4
    }
    cout << endl << "Somme : " << seq.sum() << endl; // Affiche : 10
    cout << "Maximum : " << seq.max() << endl; // Affiche : 4
    cout << "Minimum : " << seq.min() << endl; // Affiche : 1
    cout << "Produit : " << seq.prod() << endl; // Affiche : 24

    // Convertir seq en double
    Ndarray<double> seq_double = seq.astype<double>();
    cout << "Séquence (double) : ";
    seq_double.shape_str(); // Affiche : (4)
    cout << endl << "Éléments : ";
    for (size_t i = 0; i < seq_double.shape()[0]; ++i) {
        cout << seq_double({i}) << " "; // Affiche : 1 2 3 4
    }
    cout << endl << "Somme : " << seq_double.sum() << endl; // Affiche : 10.0
    cout << "Produit : " << seq_double.prod() << endl; // Affiche : 24.0

    // Tester avec un tableau de flottants
    vector<double> data = {1.5, 2.7, 3.2, 4.9};
    vector<size_t> shape = {2, 2};
    Ndarray<double> array_double(data, shape);
    cout << "Tableau personnalisé (double) : ";
    array_double.shape_str(); // Affiche : (2, 2)
    cout << endl << "Éléments : ";
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            cout << array_double({i, j}) << " "; // Affiche : 1.5 2.7 3.2 4.9
        }
    }
    cout << endl << "Somme : " << array_double.sum() << endl; // Affiche : 12.3
    cout << "Produit : " << array_double.prod() << endl; // Affiche : 305.424

    // Convertir array_double en int
    Ndarray<int> array_int = array_double.astype<int>();
    cout << "Tableau personnalisé (int) : ";
    array_int.shape_str(); // Affiche : (2, 2)
    cout << endl << "Éléments : ";
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            cout << array_int({i, j}) << " "; // Affiche : 1 2 3 4
        }
    }
    cout << endl << "Somme : " << array_int.sum() << endl; // Affiche : 10
    cout << "Produit : " << array_int.prod() << endl; // Affiche : 24

    // Tester un cas d'erreur (tableau vide)
    try {
        vector<int> empty_data = {};
        vector<size_t> empty_shape = {0};
        Ndarray<int> empty(empty_data, empty_shape);
        Ndarray<double> empty_converted = empty.astype<double>();
        cout << empty_converted.sum() << endl; // Doit lever une exception dans sum()
    } catch (const runtime_error& e) {
        cout << "Erreur : " << e.what() << endl; // Affiche : Impossible de calculer la somme d'un tableau vide.
    }

    return 0;
}