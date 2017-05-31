//
//  mesh.cpp
//  IGLFramework
//
//  Created by Apurv Nigam on 24/03/2017.
//
//

#include <stdio.h>
#include<iostream>
#include <acq/mesh.h>
#include<random>
#include<chrono>
#include "igl/invert_diag.h"
#include "thread"
#include "igl/per_vertex_normals.h"




using namespace Eigen;
using namespace std;

typedef Triplet<double> T;

Mesh::Mesh(){
    
}

void Mesh::init(const MatrixXd &vert, const MatrixXi &face){
    cout<<"Mesh Initialised "<<endl;
    
    F=face;
    V=vert;
    nV = V.rows();
    nF = F.rows();
    vNorm.resize(nV,3);
    igl::per_vertex_normals(V,F,vNorm);
    vNorm=vNorm.normalized();
    oneRingNeighbour = getOneRingNeighbours();
    uniLaplace = getUniformLaplaceMatrix();
    lbtMat.resize(nV,nV);
    areaInv.resize(nV,nV);
    
    pair2dvector = getAdjacentEdgePairs();
    areaWt = getAreaWeight();
    cotanWt = getCotanWeight();
    
}


// Find the one ring neighbours of each vertex and save them in a map.
map<int, set<int>> Mesh::getOneRingNeighbours(){
    
    map<int, set<int>> oneRingNeighbour;
    
    for(int i=0;i<nF;i++){
        Vector3i face = F.row(i);
        for (int j =0;j<face.size();j++){
            int v = face(j);
            for(int k=0;k<face.size();k++){
                if(k!= j){
                    oneRingNeighbour[v].insert(face(k));
                }
            }
        }
    }
    return oneRingNeighbour;
}

SparseMatrix<double> Mesh::getUniformLaplaceMatrix(){
    
    // Triplet (row, col, value) to hold the non-zero enteries of sparse matrix
    std::vector<T> tripletList;
    
    uniLaplace.resize(nV,nV);
    
    // There will be at max 8 elements per row in a sparse matrix of (nV X nV)
    // Preallocating space for nv X 8 triplets corresponding to nV x 8 non-zero elements, at max.
    tripletList.reserve(nV*8);
    
    for(int i=0;i<nV;i++){
        if( oneRingNeighbour.find(i)==oneRingNeighbour.end()){
            cout<<"Vertex not have any neighbour"<<endl;
            exit(0);
        }else{
            
            //Find neighbours of vertex i
            set<int> nbrs =oneRingNeighbour.at(i);
            int num = nbrs.size();
            set<int>::iterator iter;
            for(iter=nbrs.begin(); iter!=nbrs.end();++iter){
                tripletList.push_back(T(i,*iter,1.0/num));
            }
            tripletList.push_back(T(i,i,-1.0));
        }
    }
    
    //Fill the sparse matrix from the triplets
    uniLaplace.setFromTriplets(tripletList.begin(), tripletList.end());
    //Convert the sparse matrix to compressed format
    uniLaplace.makeCompressed();
    
    return uniLaplace;
    
}

vector< vector<pair<int,int>>> Mesh::getAdjacentEdgePairs(){
    pair<int,int> apair;
    vector< vector<pair<int,int> > > pair2dvector(nV, vector<pair<int,int>>(0));
    
    for(int i=0;i<nF;i++){
        Vector3i face = F.row(i);
        
        apair.first = face(1);
        apair.second = face(2);
        pair2dvector.at(face(0)).push_back(apair);
        
        apair.first = face(2);
        apair.second = face(0);
        pair2dvector.at(face(1)).push_back(apair);
        
        apair.first = face(0);
        apair.second = face(1);
        pair2dvector.at(face(2)).push_back(apair);
    }
    
    return pair2dvector;
}

SparseMatrix<double> Mesh::getAreaWeight(){
    
    areaWt.resize(nV,nV);
    std::vector<T> tripletList;
    tripletList.reserve(nV);
    
    for(int i=0;i<pair2dvector.size();i++){
        vector<pair<int,int>> pairVec = pair2dvector.at(i);
        int nPairs = pairVec.size();
        double areaSum=0;
        for(int j=0;j<nPairs;j++){
            pair<int,int> p1 = pairVec.at(j);
            int v1 = p1.first;
            int v2 = p1.second;
            Vector3d E1 =V.row(v1)-V.row(i);
            Vector3d E2 =V.row(v2)-V.row(i);
            double theta = acos( E1.dot(E2) /( E1.norm()*E2.norm() ) );
            double area = (E1.norm()*E2.norm()*sin(theta))/6.0;
            areaSum = areaSum + area;
        }
        if (areaSum==0.0) {
            cout<<"Zero Area around V"<<endl;
            exit(0);
        }
        else{
            tripletList.push_back(T(i,i,areaSum));
        }
    }
    
    //Fill the sparse matrix from the triplets
    areaWt.setFromTriplets(tripletList.begin(), tripletList.end());
    //Convert the sparse matrix to compressed format
    areaWt.makeCompressed();
    
  
    return areaWt;
}

SparseMatrix<double> Mesh::getCotanWeight(){
    
    cotanWt.resize(nV,nV);
    std::vector<T> tripletList;
    tripletList.reserve(nV*8);
    
    for(int i=0;i<pair2dvector.size();i++){
        double cotSum=0;
        vector<pair<int,int>> pairVec = pair2dvector.at(i);
        int nPairs = pairVec.size();
        for(int j=0;j<nPairs;j++){
            pair<int,int> p1 = pairVec.at(j);
            int A = p1.first;
            int C = p1.second;
            int B =-1;
            for(int j=0;j<nPairs;j++){
                if(pairVec.at(j).second == p1.first){
                    B= pairVec.at(j).first;
                }
            }
            Vector3d BO = V.row(i) - V.row(B);
            Vector3d BA = V.row(A) - V.row(B);
            Vector3d CA = V.row(A) - V.row(C);
            Vector3d CO = V.row(i) - V.row(C);
            
            double alpha = acos( BO.dot(BA) /( BO.norm()*BA.norm() ) );
            double beta = acos( CA.dot(CO) /( CA.norm()*CO.norm() ) );
            double val = 1.0/tan(alpha)+1.0/tan(beta);
            cotSum = cotSum+val;
            tripletList.push_back(T(i,A,val/2));
        }
        tripletList.push_back(T(i,i,-cotSum/2));
    }
    
    //    Fill the sparse matrix from the triplets
    cotanWt.setFromTriplets(tripletList.begin(), tripletList.end());
    //    Convert the sparse matrix to compressed format
    cotanWt.makeCompressed();

    
    return cotanWt;
    
}

MatrixXd Mesh::addNoise(double val){
    MatrixXd noisyV;
    noisyV = V;
    
    Vector3d min = V.colwise().minCoeff();
    Vector3d max = V.colwise().maxCoeff();
    double bbDLength = pow((max-min).squaredNorm(),0.5);
    double percentNoise = val;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    float mean = 0.0;
    float sDev = bbDLength*percentNoise/100;
    normal_distribution<double> distribution(mean,sDev);
    for(int i=0;i<V.rows();i++){
        for(int j=0;j<V.cols();j++){
            noisyV.row(i)(j)= V.row(i)(j)+distribution(generator);
        }
    }
    return noisyV;
}

double Mesh::calcVolume(){
    double volume=0.0;
    for(int i=0;i<nF;i++){
        Vector3i face = F.row(i);
        Vector3d v0v1 = V.row(face(1))-V.row(face(0));
        Vector3d v0v2 = V.row(face(2))-V.row(face(0));
        Vector3d fNorm = v0v1.cross(v0v2);
        Vector3d fCentroid = (V.row(face(0))+V.row(face(2))+V.row(face(2)))/3.0;
        
        volume = volume + (1/6.0)*fCentroid.dot(fNorm);
    }
    return volume;
}

MatrixXd Mesh::meanCurvature(string type){
    
    if(type=="Uniform"){
        lbtMat = uniLaplace;
    } else if(type=="Cotan"){
        igl::invert_diag(areaWt,areaInv);
        cotanWt = getCotanWeight();
        lbtMat = areaInv*cotanWt;
    }
    
    MatrixXd lapV = lbtMat*V;
    meanC.resize(nV,1);
    
    for (int i=0;i<nV;i++){
        Vector3d a= lapV.row(i);
        Vector3d b= vNorm.row(i);
        double c = a.norm()/2.0;
        double d = a.dot(b);
        
        // Sign of mean curvature is +(ve) if lapV and norm are opposite in direction
        
        if(d>=0){ //lapV and norm are in same in direction, so H is -(ve)
            c = c*-1;
        }
        meanC(i,0) = c;
        //cout<<i<<" "<<c<<endl;

    }
    
    return meanC;
}

MatrixXd Mesh::gaussCurvature(){
    
    gaussC.resize(nV,1);
    for(int i=0;i<pair2dvector.size();i++){
        vector<pair<int,int>> pairVec = pair2dvector.at(i);
        int nPairs = pairVec.size();
        double angleSum= 0;
        double areaSum=0;
        for(int j=0;j<nPairs;j++){
            pair<int,int> p1 = pairVec.at(j);
            int v1 = p1.first;
            int v2 = p1.second;
            Vector3d E1 =V.row(v1)-V.row(i);
            Vector3d E2 =V.row(v2)-V.row(i);
            double theta = acos( E1.dot(E2) /( E1.norm()*E2.norm() ) );
            angleSum = angleSum + theta;
            
            double area = (E1.norm()*E2.norm()*sin(theta))/6.0;
            areaSum = areaSum + area;
        }
        double aDeficit = (2*M_PI - angleSum);
        gaussC(i,0)=aDeficit/areaSum;
        //cout<<i<<" "<<aDeficit/areaSum<<endl;
        
    }
    
    return gaussC;
}


MatrixXd Mesh::explicitSmoothing(updateView uV, int iter){
    
    double lambda =0.00001;
    areaWt = getAreaWeight();
    
    SparseMatrix<double> eye(nV,nV);
    eye.setIdentity();
    
    int maxIter =5;
    
    
    for(int i=1;i<=maxIter;i++){
        double v0 = calcVolume();
        
        areaWt = getAreaWeight();
        
        igl::invert_diag(areaWt,areaInv);
        lbtMat = areaInv*cotanWt;
        
        V= (eye+lambda*lbtMat)*V;
        
        double vn = calcVolume();
        double beta = pow(v0/vn,1/3.0);
        V=V*beta;
        (*uV)(V);
        cout<<"Explicit Smoothing Iteration "<<i+maxIter*(iter-1)<<endl;
    }
    
    return V;
}


MatrixXd Mesh::implicitSmoothing(updateView uV, int iter){
    double lambda = 0.0001;
    int maxIter =2;
    
    for(int i=1;i<=maxIter;i++){
        double v0 = calcVolume();
        
        areaWt = getAreaWeight();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double > > solver(areaWt - lambda*cotanWt);
        assert(solver.info() == Eigen::Success);
        V = solver.solve(areaWt*V).eval();
        
        double vn = calcVolume();
        double beta = pow(v0/vn,1/3.0);
        V=V*beta;
        (*uV)(V);
        cout<<"Implicit Smoothing Iteration "<<i+maxIter*(iter-1)<<endl;
    }
    return V;
    
}

