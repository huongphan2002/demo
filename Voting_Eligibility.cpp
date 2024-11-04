//Write a program that determines whether a person is eligible to vote based on their age and citizenship status.
//Diem Huong Phan
#include <iostream>
#include <string>
using namespace std;

int main(){
    string nationality, name;
    int age;
    char UScitizenship;
    // Get information from the user
    cout << "What is your name? ";
    cin >> name;
    cout << "How old are you? ";
    cin >> age;
    cout << "What is your nationality? ";
    cin >> nationality;
    cout << "Are you citizenship?\n";
    cout << "Answer the questions with either Y for Yes or N for No. \n";
    cin >> UScitizenship;
    // Display user's information
    cout << "-----vote information-----\n";
    cout << "Name: " << name << endl;
    cout << "Age: " << age << endl;
    cout << "nationality: " << nationality << endl;
    cout << "U.S Ctizenstion: " << UScitizenship << endl;
    if (age >= 18 && (UScitizenship == 'Y')){
        cout << "You are eligible to vote.\n";
    }else{
        cout << "You are not eligible to vote.\n";
        }
    return 0;

} 