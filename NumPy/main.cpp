#include "Ndarray.hpp"
#include <iostream>
#include <vector>
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

    cout << "\n=== TESTS DES NOUVELLES MÉTHODES ===" << endl;

    // 1. Test de flatten()
    Ndarray<int> mat2D({1, 2, 3, 4}, {2, 2});
    Ndarray<int> flat = mat2D.flatten();
    cout << "\nTest flatten() : ";
    flat.shape_str(); // (4)
    cout << " -> Éléments : ";
    for (size_t i = 0; i < flat.shape()[0]; ++i) {
        cout << flat({i}) << " ";
    }
    cout << endl;

    // 2. Test de eye()
    Ndarray<int> identite = Ndarray<int>::eye(3);
    cout << "\nTest eye(3) : ";
    identite.shape_str(); // (3, 3)
    cout << "\nÉléments :\n";
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            cout << identite({i, j}) << " ";
        }
        cout << endl;
    }

    // 3. Test de linspace()
    Ndarray<double> lin = Ndarray<double>::linspace(0.0, 1.0, 5);
    cout << "\nTest linspace(0,1,5) : ";
    lin.shape_str(); // (5)
    cout << " -> Éléments : ";
    for (size_t i = 0; i < lin.shape()[0]; ++i) {
        cout << lin({i}) << " ";
    }
    cout << endl;

    // 4. Test de random() (affiche juste la forme et un élément)
    Ndarray<double> rand_arr = Ndarray<double>::random({2, 3});
    cout << "\nTest random({2,3}) : ";
    rand_arr.shape_str();
    cout << " -> Premier élément : " << rand_arr({0, 0}) << " (entre 0 et 1)" << endl;

    // 5. Statistiques avancées sur seq = [1,2,3,4]
    cout << "\nStatistiques avancées sur [1,2,3,4] :" << endl;
    cout << "Variance : " << seq.var() << endl;         // 1.25
    cout << "Écart-type : " << seq.std() << endl;       // ~1.118
    cout << "Médiane : " << seq.median() << endl;       // 2.5
    cout << "90e percentile : " << seq.percentile(90.0) << endl; // 3.7

    // 6. Opérations élément par élément
    Ndarray<int> pos = Ndarray<int>::arange(1, 4, 1); // [1,2,3]
    Ndarray<double> exp_res = pos.exp();
    Ndarray<double> log_res = pos.log();
    Ndarray<int> abs_res = pos.abs();
    Ndarray<double> pow_res = pos.pow(2.0);

    cout << "\nOpérations sur [1,2,3] :" << endl;
    cout << "exp : " << exp_res({0}) << ", " << exp_res({1}) << ", " << exp_res({2}) << endl;
    cout << "log : " << log_res({0}) << ", " << log_res({1}) << ", " << log_res({2}) << endl;
    cout << "abs : " << abs_res({0}) << ", " << abs_res({1}) << ", " << abs_res({2}) << endl;
    cout << "pow^2 : " << pow_res({0}) << ", " << pow_res({1}) << ", " << pow_res({2}) << endl;

    // 7. clip()
    Ndarray<int> to_clip({1, 5, -2, 10}, {4});
    Ndarray<int> clipped = to_clip.clip(0, 6);
    cout << "\nTest clip([1,5,-2,10], 0, 6) : ";
    for (size_t i = 0; i < 4; ++i) {
        cout << clipped({i}) << " "; // 1 5 0 6
    }
    cout << endl;

    // 8. argmin / argmax
    cout << "argmin de [1,5,-2,10] : " << to_clip.argmin() << " (valeur = " << to_clip({to_clip.argmin()}) << ")" << endl;
    cout << "argmax de [1,5,-2,10] : " << to_clip.argmax() << " (valeur = " << to_clip({to_clip.argmax()}) << ")" << endl;

    // 9. all / any
    Ndarray<int> all_true({1, 2, 3}, {3});
    Ndarray<int> some_zero({1, 0, 3}, {3});
    cout << "\nall([1,2,3]) : " << all_true.all() << endl;     // true (1)
    cout << "any([1,0,3]) : " << some_zero.any() << endl;      // true (1)
    cout << "all([1,0,3]) : " << some_zero.all() << endl;      // false (0)

    // 10. where()
    auto condition = [](int x) { return x > 2; };
    Ndarray<int> where_res = seq.where(condition, 99, -1);
    cout << "\nwhere(x>2 ? 99 : -1) sur [1,2,3,4] : ";
    for (size_t i = 0; i < 4; ++i) {
        cout << where_res({i}) << " "; // -1 -1 99 99
    }
    cout << endl;

    cout << "\n=== Tous les tests terminés ===" << endl;

    return 0;
}
