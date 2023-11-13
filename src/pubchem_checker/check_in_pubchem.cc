#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

const int N = 115330524;
// const int N = 100;
vector<string> molecules(N);


int binary_search(string str) {
    int l = 0, r = N - 1;
    while (l <= r) {
        cout << l << " " << r << "\n";
        int mid = (l + r) / 2;
        if (molecules[mid] == str)
            return mid;
        if (molecules[mid] > str)
            r = mid - 1;
        else
            l = mid + 1;
    }
    return -1;
}


int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    freopen("sorted_inchi_mols", "r", stdin);
    // freopen("inchi", "r", stdin);
    // freopen("sorted_inchi", "w", stdout);

    string str;
    int i = 0;
    while (cin >> str) {
        if (str.substr(0, 5) == "InChI") {
            molecules[i++] = str;
        }
    }
    // for (int i = 0; i < N; ++i) {
    //     cout << molecules[i] << "\n";
    // }
    vector<string> mol_list = {
        "InChI=1S/C9H17NO4/c1-7(11)14-8(5-9(12)13)6-10(2,3)4/h8H,5-6H2,1-4H3",
        "InChI=1S/C3H9NO/c1-3(5)2-4/h3,5H,2,4H2,1H3",
        "InChI=1S/C6H6O/c7-6-4-2-1-3-5-6/h1-5,7H",
        "InChI=1S/C25H16BrN3O2/c26-18-10-8-16(9-11-18)24(30)23-14-20(22-7-3-4-12-29(22)23)25(31)28-19-13-17-5-1-2-6-21(17)27-15-19/h1-15H,(H,28,31)",

        // not from pubchem
        "InChI=1S/C24H22NO4.BrH/c1-27-22-13-17(14-23(28-2)24(22)29-3)21(26)15-25-12-6-9-19-18-8-5-4-7-16(18)10-11-20(19)25;/h4-14H,15H2,1-3H3;1H/q+1;/p-1",
        "InChI=1S/C8H17.C4H8.2Na/c1-3-5-7-8-6-4-2;1-3-4-2;;/h1,3-8H2,2H3;1-4H2;;",
        "InChI=1S/C14H24ClN3O3/c1-3-4-5-6-9-20-11-21-13(19-2)10-17-12-7-8-16-14(15)18-12/h7-8,13H,3-6,9-11H2,1-2H3,(H,16,17,18)",
    };
    cout << mol_list[0] << "\n";
    cout << binary_search("InChI=1S/C9H17NO4/c1-7(11)14-8(5-9(12)13)6-10(2,3)4/h8H,5-6H2,1-4H3") << "\n";
    cout << "finished\n";
    return 0;
}