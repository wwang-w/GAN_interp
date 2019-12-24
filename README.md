# Seismic data interpolation via WGAN
data is generate by matlab using

    [Data,SegyTraceHeaders,SegyHeader]=ReadSegy(filename);
than the "Data" is cut into 1920*1152 and saved as .mat file in Matlab in the data_mat.py,the code can only handle .mat file,
Moreover the code in the [Python_segy](https://github.com/wwang-w/python_segy) can deal with the .segy file and generate patches for Network directly.
