#include <iostream>
#include <string>
#include "gdal/gdal_priv.h"
#include "resampling.hpp"

using namespace std;

int main(){

    string image_path="../examples/data/s2_clip.tif";

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

    //read the bands

    CPLErr err1= nir->RasterIO(GF_Read, 0, 0, xSize, ySize, nirData, xSize, ySize, GDT_Float32, 0, 0);
    CPLErr err2= red->RasterIO(GF_Read, 0, 0, xSize, ySize, redData, xSize, ySize, GDT_Float32, 0, 0);

    cout<<"Starting Resampling"<<endl;

    Resample resampler(xSize, ySize, geotransform[1], 5, "bc");
    cout<<xSize<<" "<<ySize<<endl;
    resampledImage resampledBand= resampler.resampleImage(nirData, "CUDA");  //resampler retruns data and dimensions of resampled data

    //now write resampled dataset as a new imagedataset
    //writing final output
    GDALDataset* outputDataset;
    GDALDriver* driver;
    driver= GetGDALDriverManager()->GetDriverByName("GTiff");
    char** opts= NULL;
    opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
    outputDataset= driver->Create("./res_bc.tif", resampledBand.xSize, resampledBand.ySize, 1, GDT_Float32, opts);
    geotransform[1]=5.0;
    geotransform[5]=-5.0;
    outputDataset->SetGeoTransform(geotransform);
    outputDataset->SetProjection(projection);

    GDALRasterBand* resampled;
    resampled= outputDataset->GetRasterBand(1);
    CPLErr err3= resampled->RasterIO(GF_Write, 0, 0, resampledBand.xSize, resampledBand.ySize, resampledBand.resampledData, resampledBand.xSize, resampledBand.ySize, GDT_Float32, 0, 0);
    GDALClose(outputDataset);
    CSLDestroy(opts);


    delete[] nirData;
    delete[] redData;
    free(resampledBand.resampledData);
    GDALClose(imageDataset);


    return 0;
}