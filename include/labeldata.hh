#ifndef LABELDATA_HH
#define LABELDATA_HH

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
//#include <vtkMatrix4x4.h>

using namespace std;

class labeldata
{
public:
    labeldata();
    labeldata(string fileName);
    void ReadFile(string fileName);
    double GetMaxDist() {return maxDist;}
    double GetMinDist() {return minDist;}
    double GetDepthFactor() {return depthFactor;}
    double* GetRotMat(int class_id, int template_id){
        if(data.find(make_pair(class_id, template_id))==data.end()){
            cerr<<"There is no label data for class"<<class_id<<" template"<<template_id<<endl;
            return nullptr;
        }//erase when confirmed
        return data[make_pair(class_id, template_id)].first;
    }
    double GetDistance(int class_id, int template_id){
        if(data.find(make_pair(class_id, template_id))==data.end()){
            cerr<<"There is no label data for class"<<class_id<<" template"<<template_id<<endl;
            return -1;
        }//erase when confirmed
        return data[make_pair(class_id, template_id)].second;
    }
private:
    double maxDist, minDist, depthFactor;
    map<pair<int, int>, pair<double*, double>> data;
    //map<pair<int, int>, pair<vtkMatrix4x4*, double>> data;
    map<int, double*> viewpoints;
};


labeldata::labeldata()
    :maxDist(2000), minDist(0), depthFactor(1)
{}
labeldata::labeldata(string fileName)
    :maxDist(2000), minDist(0), depthFactor(1)
{
    ReadFile(fileName);
}
void labeldata::ReadFile(string fileName){
    ifstream ifs(fileName);
    if(!ifs.is_open()){
        cerr<<"There is no "<<fileName<<endl;
        return;
    }
    string line, dump;
    ifs>>dump>>maxDist;
    ifs>>dump>>minDist;
    depthFactor = 2000./(maxDist-minDist) > 1? 1:2000./(maxDist-minDist);
    int class_id;
    while(getline(ifs, line)){
        stringstream ss(line);
        ss>>dump;
        if(dump=="template"){
            int template_id;
            ss>>template_id;
            data[make_pair(class_id, template_id)].first = new double[16];
            while(getline(ifs, line)){
                stringstream ss(line);
                ss>>dump;
                if(dump=="distance") ss>> data[make_pair(class_id, template_id)].second;
                else if(dump=="Elements:"){
                    ifs>>data[make_pair(class_id, template_id)].first[0]
                       >>data[make_pair(class_id, template_id)].first[1]
                       >>data[make_pair(class_id, template_id)].first[2]
                       >>data[make_pair(class_id, template_id)].first[3]
                       >>data[make_pair(class_id, template_id)].first[4]
                       >>data[make_pair(class_id, template_id)].first[5]
                       >>data[make_pair(class_id, template_id)].first[6]
                       >>data[make_pair(class_id, template_id)].first[7]
                       >>data[make_pair(class_id, template_id)].first[8]
                       >>data[make_pair(class_id, template_id)].first[9]
                       >>data[make_pair(class_id, template_id)].first[10]
                       >>data[make_pair(class_id, template_id)].first[11]
                       >>data[make_pair(class_id, template_id)].first[12]
                       >>data[make_pair(class_id, template_id)].first[13]
                       >>data[make_pair(class_id, template_id)].first[14]
                       >>data[make_pair(class_id, template_id)].first[15];

                    break;
                }
            }
        }else if(dump=="viewpoint"){
            ss>>class_id;
            viewpoints[class_id] = new double[3];
            ss>>viewpoints[class_id][0]>>viewpoints[class_id][1]>>viewpoints[class_id][2];
        }
    }ifs.close();
    cout<<"read "<<viewpoints.size()<<" classes and "<<data.size()<<" templates"<<endl;
}

#endif // LABELDATA_HH
