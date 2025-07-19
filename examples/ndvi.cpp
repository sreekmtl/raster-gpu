#include <iostream>
#include <string>
#include <dlfcn.h>
#include "gdal/gdal_priv.h"
#include "gdal/cpl_string.h"


using namespace std;

typedef void (*NormalizedIndicesFunc)(float* band1, float* band2, float* result, size_t numPixels);

void normalizedIndiceCpu(float* band1, float* band2, float* result, size_t size){
    for (size_t i=0; i<size; i++){
        result[i]= (band1[i]-band2[i])/(band1[i]+band2[i]);
    }
}

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

    double geotransform[6];
    imageDataset->GetGeoTransform(geotransform);
    const char* projection= imageDataset->GetProjectionRef();



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
    //normalizedIndiceCpu(nirData, redData, result, numPixels);


    //writing final output
    GDALDataset* outputDataset;
    GDALDriver* driver;
    driver= GetGDALDriverManager()->GetDriverByName("GTiff");
    char** opts= NULL;
    opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
    outputDataset= driver->Create("./data/ndvi2.tif", xSize, ySize, 1, GDT_Float32, opts);
    outputDataset->SetGeoTransform(geotransform);
    outputDataset->SetProjection(projection);

    GDALRasterBand* ndvi;
    ndvi= outputDataset->GetRasterBand(1);
    CPLErr err3= ndvi->RasterIO(GF_Write, 0, 0, xSize, ySize, result, xSize, ySize, GDT_Float32, 0, 0);
    GDALClose(outputDataset);
    CSLDestroy(opts);


    delete[] nirData;
    delete[] redData;
    delete[] result;
    GDALClose(imageDataset);

    return 0;



}
