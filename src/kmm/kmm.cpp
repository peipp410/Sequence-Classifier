#include <bits/stdc++.h>
#include <sys/time.h>
#include <time.h>
#include <io.h>
#include <stdlib.h>
#include <unistd.h>
using namespace std;

vector<string> reads;
vector<vector<string>> genome;
unordered_map<string, int> mp;
string file = "./data/reads.fa";

void get_reads(){
    ifstream file_reads(file);
    string seq;
    while (getline(file_reads, seq)){
        if (seq[0] != '>'){
            reads.push_back(seq);
        }
    }
}

void getAllFiles(string path, vector<string>& files, string fileType){
    long hFile = 0;
    struct _finddata_t fileinfo;
    string p;

    if ((hFile = _findfirst(p.assign(path).append("\\*" + fileType).c_str(), &fileinfo)) != -1) {
        do {
            files.push_back(p.assign(path).append("\\").append(fileinfo.name));

        } while (_findnext(hFile, &fileinfo) == 0);

        _findclose(hFile);
    }
}

void get_matrix(int k, vector<vector<vector<double>>>& p){
    int size = int(pow(4, k));
    for (int g = 0; g < 10; g++){
        string seq = genome[g][1];
        //vector<vector<double>> prob(size, vector<double>(4, 0));
        for (int i = 0; i <= seq.size()-k-1; i++){
            int pos = strtol(seq.substr(i, k).c_str(), nullptr, 4);
            int curr = seq[i+k]-'0';
            if (curr < 0 || curr > 3) continue;
            p[g][pos][curr]++;
        }
        //cout<<p[0][0][0]<<endl;
        for (int i = 0; i < size; i++){
            for (int j = 0; j < 4; j++){
                p[g][i][j] = p[g][i][j]/(seq.size()-k);
            }
        }
    }

}

void get_genomes(string reference){
    //vector<string> filename = {"007984", "008709", "009511", "009767", "011126", "011138", "013943", "015656", "015722", "015859"};
    vector<string> filename;
    getAllFiles(reference, filename, ".fna");
    for (string& s : filename){
        //s = "./data/genomes/NC_" + s + ".fna";
        ifstream file_genom(s);
        string seq, name, genom_seq;
        while(getline(file_genom, seq)){
            if (seq[0] == '>'){
                int pos = seq.find("| ", 0);
                name = seq.substr(pos+2);
            }
            else{
                if (seq[seq.size()-1] == '\n'){
                    seq.erase(seq.size()-1);
                }
                genom_seq += seq;
            }
        }
        for (int i = 0; i < genom_seq.size(); i++){
            if (genom_seq[i] == 'A') genom_seq[i] = '0';
            else if (genom_seq[i] == 'G') genom_seq[i] = '1';
            else if (genom_seq[i] == 'C') genom_seq[i] = '2';
            else if (genom_seq[i] == 'T') genom_seq[i] = '3';
            else{
                genom_seq[i] = rand()%4 + '0';
            }
        }
//        replace(genom_seq.begin(), genom_seq.end(), 'A', '0');
//        replace(genom_seq.begin(), genom_seq.end(), 'G', '1');
//        replace(genom_seq.begin(), genom_seq.end(), 'C', '2');
//        replace(genom_seq.begin(), genom_seq.end(), 'T', '3');
        vector<string> tmp;
        tmp.push_back(name);
        tmp.push_back(genom_seq);
        genome.push_back(tmp);
    }
    for (int i = 0; i < 10; i++){
        mp[genome[i][0]] = i;
    }
}

void reads_score(vector<vector<vector<double>>>& p, vector<double>& score, vector<int>& group, int k){
    for (int m = 0; m < reads.size(); m++){
        string seq = reads[m];
        for (int i = 0; i < seq.size(); i++){
            if (seq[i] == 'A') seq[i] = '0';
            else if (seq[i] == 'G') seq[i] = '1';
            else if (seq[i] == 'C') seq[i] = '2';
            else if (seq[i] == 'T') seq[i] = '3';
            else{
                seq[i] = rand()%4 + '0';
            }
        }
//        replace(seq.begin(), seq.end(), 'A', '0');
//        replace(seq.begin(), seq.end(), 'G', '1');
//        replace(seq.begin(), seq.end(), 'C', '2');
//        replace(seq.begin(), seq.end(), 'T', '3');
        vector<double> temp_score(10);
        for (int j = 0; j < 10; j++){
            double s = 0;
            for (int i = 0; i < seq.size()-k-1; i++){
                int pos = strtol(seq.substr(i, k).c_str(), nullptr, 4);
                int curr = seq[i+k]-'0';
                if (curr < 0 || curr > 3) continue;
                s -= log(p[j][pos][curr]);
            }
            temp_score[j] = s;
        }
        double min_score = 10000000000;
        int min_pos = 0;
        for (int i = 0; i < 10; i++){
            if (temp_score[i] < min_score){
                min_score = temp_score[i];
                min_pos = i;
            }
        }
        score[m] = min_score;
        group[m] = min_pos;
    }
}

void score_stat(vector<int>& group, int k){
    vector<int> stat(10);
    for (int& g : group){
        stat[g]++;
    }
    printf("------------------------\n");
    printf("k=%d\n", k);
    for (int i = 0; i < 10; i++){
        printf("Group %d: %d\n", i, stat[i]);
    }
}

double accuracy(vector<int>& group, string label){
    ifstream file_reads(label);
    string seq;
    int i = 0, correct = 0;
    while (getline(file_reads, seq)){
        if (group[i] == mp[seq.substr(seq.find('\t')+1)]){
            correct++;
        }
        i++;
    }
    return correct*1.0/(i+1);
}

//double get_wall_time(){
//    struct timeval time ;
//    if (gettimeofday(&time,NULL)){
//        return 0;
//    }
//    return (double)time.tv_sec + (double)time.tv_usec * .000001;
//}

int main(int argc, char** argv){
    bool test;
    int opt, k;
    string reference, label;
    // if (argc == 5) test = 1;
    // else if (argc < 4 || argc > 5){
    //     cout<<"Usage: .\\kmm.exe k reference_fold short_genomes [label]"<<endl;
    //     return 0;
    // }

    while((opt=getopt(argc,argv,"k:r:g:t:h")) != -1){
        switch(opt){
            case 'h':
                cout<<"Usage: .\\kmm.exe -k k -r reference_fold -g short_genomes [-t label]"<<endl;
                return 0;
            case 'k':
                k = atoi(optarg);
                break;
            case 'r':
                reference = optarg;
                break;
            case 'g':
                file = optarg;
                break;
            case 't':
                test = 1;
                label = optarg;
        }
    }

    srand((unsigned)time(NULL));
    get_reads();
    get_genomes(reference);
    clock_t start, end;
    start = clock();
    int size = int(pow(4, k));
    vector<vector<vector<double>>> p(10, vector<vector<double>>(size, vector<double>(4, 0)));
    get_matrix(k, p);
    int tot_reads = reads.size();
    vector<double> score(tot_reads);
    vector<int> group(tot_reads);
    reads_score(p, score, group, k);
    score_stat(group, k);
    if (test){
        cout<<"accuracy: "<<accuracy(group, label)<<endl;
    }
    end = clock();
    cout<<"Running time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
    return 0;
}