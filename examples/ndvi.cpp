#include <iostream>
#include <string>
#include <dlfcn.h>
#include "gdal/gdal_priv.h"


using namespace std;

typedef void (*NormalizedIndicesFunc)(float* band1, float* band2, float* result, size_t numPixels);

int main(){

    string image_path="./data/sample.tif";

    GDALAllRegister();
    GDALDataset* imageDataset;

    
    imageDataset=(GDALDataset*)GDALOpen(image_path.c_str(), GA_ReadOnly);

    if (imageDataset==nullptr){
        cerr<<"Error in opening the data";
    }

    int xSize= imageDataset->GetRasterXSize();
    int ySize= imageDataset->GetRasterYSize();

    size_t numPixels= static_cast<size_t>(xSize*ySize);

    GDALRasterBand *nir= imageDataset->GetRasterBand(2);
    GDALRasterBand *red= imageDataset->GetRasterBand(3);

    float* nirData= new float[numPixels](); //band1
    float* redData= new float[numPixels](); //band2
    float* result= new float[numPixels]();  //result

    //read the bands

    CPLErr err1= nir->RasterIO(GF_Read, 0, 0, xSize, ySize, nirData, xSize, ySize, GDT_Float32, 0, 0);
    CPLErr err2= red->RasterIO(GF_Read, 0, 0, xSize, ySize, redData, xSize, ySize, GDT_Float32, 0, 0);

    //load the function from shared library and compute indices
    //Accessing function at run time from a shared object

    void* handle= dlopen("../map_algebra/libmapalg.so", RTLD_LAZY);
    if (!handle){
        cerr<<"Failed to load .so"<<dlerror()<<endl;
        return 1;
    }
    dlerror();
    NormalizedIndicesFunc normalized_indices= (NormalizedIndicesFunc)dlsym(handle, "normalized_indices");
    const char* dlsym_error= dlerror();
    if (dlsym_error){
        cerr<<"Cannot load symbol normalized_indices: "<<dlsym_error<<endl;
        dlclose(handle);
        return 1;
    }

    normalized_indices(nirData, redData, result, numPixels);


    //print first 100 pixel values
    for (size_t i=0; i<1000; i++){
        cout<<result[i]<<" ";
    }

    delete[] nirData;
    delete[] redData;
    delete[] result;
    GDALClose(imageDataset);

    return 0;



}
