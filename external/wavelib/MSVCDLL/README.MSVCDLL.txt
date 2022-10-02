1. I have included both Debug and Release builds as MSVC++ has issues mixing up Release and Debug.

2. Required Files - wavelet2d.h, wavelet2d.lib ,wavelet2d.dll AND libfftw3-3.dll (included in the fftw3 folder. Alternatively, you may want to install fftw3 library from www.fftw.org. I have used version 3.2.2 in this particular case).

3. You will need to add wavelet2d.lib to the additional dependency in your project. Essentially, this is the
 file that you are going to link to and it will open wavelet2d.dll provided it is in the path.Make sure that libfftw3-3.dll
is also included in one of the paths.

[Ignore 4 and 5 if you are used to working with DLLs in MS Visual Studio. Your way is the already the best one]
 
4.To learn more about windows and DLL search path visit- http://msdn.microsoft.com/en-us/library/7d83bc18(VS.71).aspx 

5.You can't mix release and debug versions. Both versions are named wavelet2d.dll so putting them in system path is not a good idea either. Preferred way is to use Visual Studio project settings to link to the wavelet2d.lib file and put the release and debug wavelet2d dlls along with libfftw-3.3dll in the directory containing the respective release and debug executables of your project. Import library will open the DLLs.

Date Modified- 08/17/2011