#include <igl/read_triangle_mesh.h>
#include <igl/combine.h>
#include <igl/viewer/Viewer.h>
#include <Eigen/Geometry>
#include <thread>
#include <string>
#include <igl/jet.h>

#include "acq/mesh.h"

using namespace std;
using namespace Eigen;

int numMesh=1;

vector<MatrixXd> vecV(numMesh);
vector<MatrixXi> vecF(numMesh);

igl::viewer::Viewer viewer;
MatrixXd V,colorMap,vOriginal;
MatrixXi F;
SparseMatrix<double> cotMat;
MatrixXd normP;
Mesh mesh;
int iter=0;


//Call back function called during smoothing to show updated mesh after every iteration
void updateView1(MatrixXd V){
    //  cout<<"Update View"<<endl;
    
    igl::jet(mesh.gaussCurvature(),true,colorMap);
    
//    igl::per_vertex_normals(V,F,normP);
//    colorMap = normP.rowwise().normalized().array()*0.5+0.5;
//    
    viewer.data.compute_normals();
    
    
    viewer.data.set_colors(colorMap);
    viewer.data.set_vertices(V);
}

int main(int argc, char * argv[])
{
    string meshPath;
    //meshPath="./cow.off";
    //meshPath = "./bumpy.off";
    // meshPath = "./cheburashka.off";
    //meshPath = "./decimated-knight.off";
    //meshPath = "./bunny.off";
    
    meshPath = argv[1];
    igl::read_triangle_mesh(meshPath,V,F);
    mesh.init(V,F);
    
    vOriginal=V;
 //   normP.resize(V.rows(),3);

    //code to add noise
    //V = mesh.addNoise(0.5);
    //mesh.init(V,F);
    
        igl::jet(mesh.gaussCurvature(),true,colorMap);
    viewer.data.set_mesh(V,F);
    
    //Code for Mean Curvature
//        MatrixXd meanC = mesh.meanCurvature("Cotan");
//        igl::jet(meanC,true,colorMap);
    
    //Code for Gaussian Curvature
//        MatrixXd gaussC = mesh.gaussCurvature();
//        igl::jet(gaussC,true,colorMap);
    
    
    const auto &key_down = [](igl::viewer::Viewer &viewer,unsigned char key,int mod)->bool
    {
        switch(key)
        {
            case 'r':
            case 'R':
                V = vOriginal;
                mesh.init(V,F);
                iter=0;
                igl::jet(mesh.gaussCurvature(),true,colorMap);
                viewer.data.set_colors(colorMap);
                viewer.data.set_vertices(V);
                
                break;
            case 'B':
            {
                iter=iter+1;
                thread t1(&Mesh::implicitSmoothing, &mesh,updateView1,iter);
                t1.detach();
                
                break;
            }
            case 'E':
            {
                iter=iter+1;
                thread t1(&Mesh::explicitSmoothing, &mesh,updateView1,iter);
                t1.detach();
                
                break;
            }
                
            default:
                return false;
        }
        //viewer.data.set_vertices(V);
        return true;
    };
    
    viewer.data.set_colors(colorMap);
    viewer.core.is_animating=true;
    viewer.callback_key_down = key_down;
    viewer.launch();
    
}

