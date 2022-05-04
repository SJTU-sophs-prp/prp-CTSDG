#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main() {

    string target;
    string train_or_test;

    cout <<"This code is specifically designed to run CTSDG commands under the code directory" << endl;

    ofstream out("trainer.sh");
    cout << "which card would you like to run on?" << endl;
    cin >> target;
    out << "CUDA_VISIBLE_DEVICES=" << target << " python ";
    cout << "train or test? 1 for train and 2 for test" << endl;
    cin >> train_or_test;
    if(train_or_test == "1")
    {
        //1 for train
        out << "train.py ";
        cout <<"image size?" << endl;
        cin >> target;
        if(target == "128")
        {
            out << "--load_size 128 ";
            cout << "be aware that you are using size 128" << endl;
            cout << "Which dataset to choose? [21123 master_data]" << endl;
            cout << "                                  1" << endl;
            cin >> target;
            out << "--image_root ~/code/CTSDG-data/datasets/128/ ";

        }
        else if(target == "256")
        {
            out << "--load_size 256 ";
            cout << "be aware that you are using size 256" << endl;
            cout << "Which dataset to choose? [21123 master_data | 5196 sswy256]" << endl;
            cout << "                                    1                 2" << endl;
            cin >> target;
            if(target=="1")
            {
                out << "--image_root ~/code/CTSDG-data/datasets/master_dataset/ ";
            }
            else if(target=="2")
            {
                out << "--imagr_root ~/code/huangjiawei/sswy256/ ";
            }
        }
        cout << "please input the mask you like" << endl;
        cin >> target;
        out << "--mask_root ~/code/CTSDG-data/masks/" << target << "/ ";
        cout << "please input the save location" << endl;
        cin >> target;
        out << "--save_dir ~/code/CTSDG-data/masks/ckpt/" << target << "/ ";







    }
    else
    {
        out << "test.py ";
        cout << "image size?" << endl;
        cin >> target;
        if(target == "128")
        {
            out << "--load_size 128 --image_root ~/code/CTSDG/combine_contrast_128/ ";
            cout << "be aware that you are using size 128" << endl;
        }
        else if(target == "256")
        {
            out << "--load_size 256 --image_root ~/code/CTSDG/combined_contrast_cali/ ";
            cout << "be aware that you are using size 256" << endl;
        }
        cout << "please input your mask root" << endl;
        cin >>target;
        out << "--mask_root ~/code/CTSDG-data/masks/" << target << "/ ";
        cout << "pre trained model?" << endl;
        cin >> target;
        out << "--pre_trained ~/code/CTSDG-data/masks/ckpt/" << target << " ";
        cout << "result root?" << endl;
        cin >> target;
        out << "--result_root ./result/" << target << "/ ";
        cout << "number of evaluation?" << endl;
        cin >> target;
        out << "--number_eval " << target << " ";

    }


    target = "daiuwsd";
    cout << "is there anything you would like to add?" << endl;

    getline(cin, target);
    while(target != "n")
    {
        out <<" " << target;

        cout << "enter the command straight forward or end with a n" << endl;
        getline(cin, target);


    }

    cout << "all set and ready to go!" << endl;

    return 0;
}
