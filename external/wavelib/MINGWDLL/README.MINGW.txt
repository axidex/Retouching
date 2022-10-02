1. I have included both Debug and Release builds.

2. Required Files - wavelet2d.h, wavelet2d.lib ,wavelet2d.dll AND libfftw3-3.dll. libfftw3-3.dll.a is also included. It is recommended that you link to the Release and Debug folders as they are. Once you link to import library and specify the folder location, wavelet import library should automatically open the two DLLs. 

3. GNU-GCC compiler is used so mixing debug and release versions shouldn't be an issue.

Date Modified- 08/17/2011