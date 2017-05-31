//
//  mesh.h
//  IGLFramework
//
//  Created by Apurv Nigam on 24/03/2017.
//
//

#ifndef mesh_h
#define mesh_h

#include <stdio.h>
#include "Eigen/Eigenvalues"
#include <Eigen/Sparse>
#include <set>
#include<random>
#include<chrono>

using namespace std;

//Typedef for a callback function for updating the viewer
typedef void (*updateView)(Eigen::MatrixXd);


/*
 
 Definition class Mesh. It hold the important data structures related to a triangular mesh
 and functions to calculate , mean and gaussioan curvature, perform explicit and implicit 
 smoothing and calculate mesh volume.
 
 */

class Mesh{
    //Faces of he mesh
    Eigen::MatrixXi F;
    
    //Vertices of the mesh
    Eigen::MatrixXd V;
    
    //Vertex normals of the mesh
    Eigen::MatrixXd vNorm;
    
    //No. of vertices in mesh
    size_t  nV;
    
    //Np. of faces in the mesh
    size_t nF;
    
    //Mean Curvature of the mesh
    Eigen::MatrixXd meanC;
    
    //Gaussian Curvature of the mesh
    Eigen::MatrixXd gaussC;
    
    /*
     Data structure to hold the pair of edges going  of each vertex in the mesh
     Vertices that form pair are from same face.
     Corresponding to each vertex, we save all such pairs in a vector. 
     We save N such vectors,ceach corresponding to a vertex

     
     V1 | [Va Vb]  [Vc Vd] . . . [Vi Vj] |
     V2 |                                |
     V3 |                                |
      . |                                |
      . |                                |
      . |                                |
     Vn | [Vp Vq]  [Vp Vr] . . . [Vs Vc] |
     
     a,b,c, ... z are arbitary numbers in range 1 to N */
    vector< vector<pair<int,int> > > pair2dvector;

    // Map to hold one ring neighbour of each vertex
    // If a face in a tri mesh has vertices ( v1, v2, v3), then
    // v2 and v3 are one ring neighbours for v1,
    // v1 and v3 are one ring neighbour for v2 and
    // v1 and v2 are one ring neighbour for v3.
    // We iterarte through each face and collect the one ring neighbours.
    // We save them in a set to take care of duplicate enteries
    std::map<int ,std::set<int> >oneRingNeighbour;
    
    // Laplace Beltrami Matrix using uniform discretization
    Eigen::SparseMatrix<double> uniLaplace;
    
    // Laplace Matrix
    //In case of Uniform Discretization, it is uniLaplace, else areaInv*cotanWt
    Eigen::SparseMatrix<double> lbtMat;
    
    //Cotan Weights for the mesh
    Eigen::SparseMatrix<double> cotanWt;
    
    //Area weights around each vertex in the mesh.
    Eigen::SparseMatrix<double> areaWt;
    
    //Area inverse weights around each vertex in the mesh.
    Eigen::SparseMatrix<double> areaInv;
    
    
public:
    //Default constructor
    Mesh();
    
    // Function to init private members of mesh
    void init(const Eigen::MatrixXd &vert, const Eigen::MatrixXi &face);
    
    void setVertices(const Eigen::MatrixXd &vert);
    //Construct the data structure to hold one ring neighbours
    std::map<int, std::set<int>> getOneRingNeighbours();
    
    //Construct the uniform Laplace-Beltramin Matrix
    Eigen::SparseMatrix<double> getUniformLaplaceMatrix();
    
    // Contruct the data structure holding the edge pairs coming out from vertex
    vector<vector<pair<int,int>>> getAdjacentEdgePairs();
    
    //Contruct the area weight matrix that hols area around each vertex
    Eigen::SparseMatrix<double> getAreaWeight();
    
    //Calculate the cotan weights of mesh
    Eigen::SparseMatrix<double> getCotanWeight();
    
    //Add noise to mesh in proportion to bounding box volume
    Eigen::MatrixXd addNoise(double val);
    
    //Calculate the volume of the mesh
    double calcVolume();
    
    //Calculate the mean curvature of the mesh, according to type of
    //discretization passed as function parameter
    Eigen::MatrixXd meanCurvature(string type);
    
    //Calculate the gausian curvature of the mesh
    Eigen::MatrixXd gaussCurvature();
    
    //Perform implicit smoothing of the mesh
    Eigen::MatrixXd implicitSmoothing(updateView uV,int iter);
    
    //Perform explicit smoothing of the mesh
    Eigen::MatrixXd explicitSmoothing(updateView uV, int iter);
    
};
#endif /* mesh_h */
