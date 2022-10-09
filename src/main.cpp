/**
 * @file BinarySkinMask.cpp
 * @brief An example of how to create a binary skin mask from face landmark locations.
 *
 */
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/ml.hpp>
#include <wavelib.h>

#include <ranges>
#include <tinysplinecxx.h>
#include <iostream>

#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>


void GMM(cv::Mat src, cv::Mat& dst) {
    // Define 5 colors with a maximum classification of no more than 5
    cv::Scalar color_tab[] = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0),
                              cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255),
                              cv::Scalar(255, 0, 255) };

    int width = src.cols;
    int height = src.rows;
    int dims = src.channels();

    int nsamples = width * height;
    cv::Mat points(nsamples, dims, CV_64FC1);
    cv::Mat labels;
    cv::Mat result = cv::Mat::zeros(src.size(), CV_8UC3);

    // Define classification, that is, how many classification points of function K value
    int num_cluster = 3;
    printf("num of num_cluster is %d\n", num_cluster);
    // Image RGB pixel data to sample data
    int index = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            index = row * width + col;
            cv::Vec3b rgb = src.at<cv::Vec3b>(row, col);
            points.at<double>(index, 0) = static_cast<int>(rgb[0]);
            points.at<double>(index, 1) = static_cast<int>(rgb[1]);
            points.at<double>(index, 2) = static_cast<int>(rgb[2]);
        }
    }

    // EM Cluster Train
    cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
    // Partition number
    em_model->setClustersNumber(num_cluster);
    // Set covariance matrix type
    em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
    // Set convergence conditions
    em_model->setTermCriteria(
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.1));
    // Store the probability partition to labs EM according to the sample training
    em_model->trainEM(points, cv::noArray(), labels, cv::noArray());

    // Mark color and display for each pixel
    cv::Mat sample(1, dims, CV_64FC1); //
    int r = 0, g = 0, b = 0;
    // Put each pixel in the sample
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            index = row * width + col;

            // Get the color of each channel
            b = src.at<cv::Vec3b>(row, col)[0];
            g = src.at<cv::Vec3b>(row, col)[1];
            r = src.at<cv::Vec3b>(row, col)[2];

            // Put pixels in sample data
            sample.at<double>(0, 0) = static_cast<double>(b);
            sample.at<double>(0, 1) = static_cast<double>(g);
            sample.at<double>(0, 2) = static_cast<double>(r);

            // Rounding
            int response = cvRound(em_model->predict2(sample, cv::noArray())[1]);
            cv::Scalar c = color_tab[response];
            result.at<cv::Vec3b>(row, col)[0] = c[0];
            result.at<cv::Vec3b>(row, col)[1] = c[1];
            result.at<cv::Vec3b>(row, col)[2] = c[2];
        }
    }
    cv::imshow("gmm", result);
    cv::imwrite("gmm.jpg", result);
}

double absmax(double* array, int N) {
    double max;
    int i;

    max = 0.0;
    for (i = 0; i < N; ++i) {
        if (fabs(array[i]) >= max) {
            max = fabs(array[i]);
        }
    }

    return max;
}

void medianaa(double* arr1, double* arr2, int row, int col)
{
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            arr2[i * col + j] = (arr1[i * col + j] + arr2[i * col + j]) / 2;
            //std::cout  << i << " " << j << " " << row << " " << col <<" " << arr1[i * col + j] << std::endl;
        }
    }
}

/// @brief Bianry skin mask example
///
/// Usage: BinarySkinMask.exe [params] image landmark_model
int main(int argc, char** argv) {
    
    std::string imgDir = "1.jpg";
    std::string modelDir = "shape_predictor_68_face_landmarks.dat";
    const auto inputImg =
        cv::imread(cv::samples::findFile(imgDir, /*required=*/false, /*silentMode=*/true));
    if (inputImg.empty()) {
        std::cout << "Could not open or find the image: " << imgDir << "\n"
            << "The image should be located in `images_dir`.\n";
        return -1;
    }
    // Make a copy for drawing landmarks
    cv::Mat landmarkImg = inputImg.clone();
    // Make a copy for drawing binary mask
    cv::Mat maskImg = cv::Mat::zeros(inputImg.size(), CV_8UC1);


    auto landmarkModelPath = cv::samples::findFile(modelDir, /*required=*/false);
    if (landmarkModelPath.empty()) {
        std::cout << "Could not find the landmark model file: " << modelDir << "\n"
            << "The model should be located in `models_dir`.\n";
        return -1;
    }

    ////////////////////////////////////////////////////// ������

    // Leave the original input image untouched
    cv::Mat workImg = inputImg.clone();
    
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize(landmarkModelPath) >> landmarkDetector;

    // Detect faces
    // Need to use `dlib::cv_image` to bridge OpenCV and dlib.
    const auto dlibImg = dlib::cv_image<dlib::bgr_pixel>(inputImg);
    auto faceDetector = dlib::get_frontal_face_detector();
    auto faces = faceDetector(dlibImg);

    // Draw landmark on the input image
    const auto drawLandmark = [&](const auto x, const auto y) {
        constexpr auto radius = 5;
        const auto color = cv::Scalar(0, 255, 255);
        constexpr auto thickness = 2;
        const auto center = cv::Point(x, y);
        cv::circle(landmarkImg, center, radius, color, thickness);
    };

    // clang-format off
    // Get outer contour of facial features
    // The 68 facial landmark from the iBUG 300-W dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
    // Jaw:              0-16 (lower face boundary)
    // Right eyebrow:   17-21
    // Left eyebrow:    22-26 
    // Nose:            27-35
    // Right eye:       36-41
    // Left eye:        42-47
    // Mouth:           48-67 (boundary:48-59)
    // clang-format on
    const auto jaw = [](const auto i) { return i >= 0 && i <= 16; };
    const auto rightEye = [](const auto i) { return i >= 36 && i <= 41; };
    const auto leftEye = [](const auto i) { return i >= 42 && i <= 47; };
    const auto mouthBoundary = [](const auto i) { return i >= 48 && i <= 59; };

    const auto detect = [&](const auto& face) { return landmarkDetector(dlibImg, face); };
    for (const auto& shape : faces | std::views::transform(detect)) {
        std::vector<tinyspline::real> knots;
        // Join the landmark points on the boundary of facial features using cubic curve
        const auto getCurve = [&]<typename T>(T predicate, const auto n) {
            knots.clear();
            for (const auto i :
                std::views::iota(0) | std::views::filter(predicate) | std::views::take(n)) {
                const auto& point = shape.part(i);
                knots.push_back(point.x());
                knots.push_back(point.y());
            }
            // Make a closed curve
            knots.push_back(knots[0]);
            knots.push_back(knots[1]);
            // Interpolate the curve
            auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);
            return spline;
        };

        // Right eye cubic curve
        constexpr auto nEyeCurve = 6;
        const auto rightEyeCurve = getCurve(rightEye, nEyeCurve);
        // Sample landmark points from the curve
        constexpr auto eyePointNum = 25;
        std::array<cv::Point, eyePointNum> rightEyePts;
        for (const auto i : std::views::iota(0, eyePointNum)) {
            const auto net = rightEyeCurve(1.0 / eyePointNum * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            drawLandmark(x, y);
            rightEyePts[i] = cv::Point(x, y);
        }
        // Draw binary mask
        cv::fillConvexPoly(maskImg, rightEyePts, cv::Scalar(255), cv::LINE_AA);


        // Left eye cubic curve
        const auto leftEyeCurve = getCurve(leftEye, nEyeCurve);
        std::array<cv::Point, eyePointNum> leftEyePts;
        // Sample landmark points from the curve
        for (const auto i : std::views::iota(0, eyePointNum)) {
            const auto net = leftEyeCurve(1.0 / eyePointNum * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            drawLandmark(x, y);
            leftEyePts[i] = cv::Point(x, y);
        }
        // Draw binary mask
        cv::fillConvexPoly(maskImg, leftEyePts, cv::Scalar(255), cv::LINE_AA);

        // Mouth cubic curve
        constexpr auto nMouthCurve = 12;
        const auto mouthCurve = getCurve(mouthBoundary, nMouthCurve);
        constexpr auto mouthPointNum = 40;
        std::array<cv::Point, mouthPointNum> mouthPts;
        // Sample landmark points from the curve
        for (const auto i : std::views::iota(0, mouthPointNum)) {
            const auto net = mouthCurve(1.0 / mouthPointNum * i);
            const auto result = net.result();
            const auto x = result[0], y = result[1];
            drawLandmark(x, y);
            mouthPts[i] = cv::Point(x, y);
        }
        // Draw binary mask
        cv::fillPoly(maskImg, mouthPts, cv::Scalar(255), cv::LINE_AA);

        // Estimate an ellipse that can complete the upper face region
        constexpr auto nJaw = 17;
        std::vector<cv::Point> lowerFacePts;
        for (auto i : std::views::iota(0) | std::views::filter(jaw) | std::views::take(nJaw)) {
            const auto& point = shape.part(i);
            const auto x = point.x(), y = point.y();
            drawLandmark(x, y);
            lowerFacePts.push_back(cv::Point(x, y));
        }
        // Guess a point located in the upper face region
        // Pb: 8 (bottom of jaw)
        // Pt: 27 (top of nose
        const auto& Pb = shape.part(8);
        const auto& Pt = shape.part(27);
        const auto x = Pb.x();
        const auto y = Pt.y() - 0.85 * abs(Pb.y() - Pt.y());
        drawLandmark(x, y);
        lowerFacePts.push_back(cv::Point(x, y));
        // Fit ellipse
        const auto box = cv::fitEllipseDirect(lowerFacePts);
        cv::Mat maskTmp = cv::Mat(maskImg.size(), CV_8UC1, cv::Scalar(255));
        cv::ellipse(maskTmp, box, cv::Scalar(0), /*thickness=*/-1, cv::FILLED);

        cv::bitwise_or(maskTmp, maskImg, maskImg);
        cv::bitwise_not(maskImg, maskImg);
    }


    cv::Mat maskChannels[3] = { maskImg, maskImg, maskImg };
    cv::Mat maskImg3C;
    cv::merge(maskChannels, 3, maskImg3C);
    cv::Mat spotImg, spotImgT;
    cv::Mat maskImgNot, maskGF;

    //int a;
    //double b;
    //std::cin >> a;
    //std::cin >> b;
    cv::bitwise_and(inputImg, maskImg3C, spotImgT);



    cv::ximgproc::guidedFilter(spotImgT, maskImg3C, maskGF, 10, 200); //10 200
    cv::bitwise_not(maskGF, maskImgNot);

    cv::bitwise_and(inputImg, maskGF, spotImg);

    cv::Mat t;
    //bilateralFilter(maskGF, t, dx, fc, fc);
    //t.copyTo(maskGF);
    cv::imshow("maskGF", maskGF);



    cv::Mat tmp1, tmp2, noFace;


    // Inner mask
    cv::Mat maskEx;
    cv::Mat maskElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)); //71 71
    //cv::imshow("maskElement", maskElement);
    cv::morphologyEx(maskImg, maskEx, cv::MORPH_ERODE, maskElement);
    cv::Mat maskExs[3] = {maskEx, maskEx, maskEx};
    cv::Mat maskEx3C;
    cv::merge(maskExs, 3, maskEx3C);
    
     // Make a preserved image for future use
    cv::Mat preservedImg, maskPres;
    cv::bitwise_not(maskEx3C, maskPres);
    cv::bitwise_and(workImg, maskPres, preservedImg);
    
    // Spot Concealment
    // Convert the RGB image to a single channel gray image
    cv::Mat grayImg;
    cv::cvtColor(workImg, grayImg, cv::COLOR_BGR2GRAY);

    // Compute the DoG to detect edges
    cv::Mat blurImg1, blurImg2, dogImg;
    const auto sigmaY = grayImg.cols / 200.0;
    const auto sigmaX = grayImg.rows / 200.0;
    cv::GaussianBlur(grayImg, blurImg1, cv::Size(3, 3), /*sigma=*/0);
    cv::GaussianBlur(grayImg, blurImg2, cv::Size(0, 0), sigmaX, sigmaY);
    cv::subtract(blurImg2, blurImg1, dogImg);
    cv::Mat not_dogImg;
    cv::bitwise_not(dogImg, not_dogImg);
    cv::imshow("not_dogImg", not_dogImg);
    // Apply binary mask to the image
    cv::Mat not_dogImgs[3] = {not_dogImg, not_dogImg, not_dogImg};
    cv::Mat not_dogImg3C;
    cv::merge(not_dogImgs, 3, not_dogImg3C);

    cv::Mat final_mask, final_mask_not;
    cv::bitwise_and(maskGF, not_dogImg3C, final_mask);
    cv::bitwise_not(final_mask, final_mask_not);
    cv::imshow("final_mask", final_mask);
    cv::imshow("final_mask_not", final_mask_not);
    cv::Mat final_face_not, final_face;
    cv::bitwise_and(workImg, final_mask, final_face);
    cv::bitwise_and(workImg, final_mask_not, final_face_not);

    spotImg = final_face.clone();
    /*int value1, value2;
    std::cin >> value1;
    std::cin >> value2;
    int dx = value1 * 5;
    double fc = value1 * 12.5;
    */
    int dx = 5;
    double fc = 50;
    //std::cin >> dx;
    //std::cin >> fc;
    bilateralFilter(spotImg, tmp1, dx, fc, fc);

    //cv::Mat gmm;
    //GMM(inputImg, gmm);


    cv::bitwise_and(inputImg, maskImgNot, noFace);
    cv::imshow("noFace", noFace);


    cv::add(final_face_not, tmp1, tmp2);
    cv::Mat dst;
    bilateralFilter(tmp2, dst, 5, 20, 20);
    cv::imshow("dst", dst);
    cv::imshow("workImg", workImg);
    cv::imwrite("final.jpg", dst);
  
    

    
    ////////////////////////// Spliting into 1 color

    cv::Mat bgrchannel_smoothed[3], bgrchannel_orig[3];

    cv::split(dst.clone(), bgrchannel_smoothed);
    cv::split(workImg.clone(), bgrchannel_orig);

    cv::Mat blue_double_smoothed, green_double_smoothed, red_double_smoothed, blue_double_orig, green_double_orig, red_double_orig;

    bgrchannel_smoothed[0].convertTo(blue_double_smoothed, CV_64F);
    bgrchannel_smoothed[1].convertTo(green_double_smoothed, CV_64F);
    bgrchannel_smoothed[2].convertTo(red_double_smoothed, CV_64F);

    bgrchannel_orig[0].convertTo(blue_double_orig, CV_64F);
    bgrchannel_orig[1].convertTo(green_double_orig, CV_64F);
    bgrchannel_orig[2].convertTo(red_double_orig, CV_64F);

    double* red_smoothed = red_double_smoothed.ptr<double>(0);
    double* green_smoothed = green_double_smoothed.ptr<double>(0);
    double* blue_smoothed = blue_double_smoothed.ptr<double>(0);

    double* red_orig = red_double_orig.ptr<double>(0);
    double* green_orig = green_double_orig.ptr<double>(0);
    double* blue_orig = blue_double_orig.ptr<double>(0);

    ////////////////////////// WAVELET

    // BLUE
    wave_object obj_blue_smoothed, obj_blue_orig;
    wt2_object wt_blue_smoothed, wt_blue_orig;
    
    int J = 3;
    double * wavecoeffs_blue_smoothed, * wavecoeffs_blue_orig, * oup_blue;
    double* cHH1_blue_smoothed, * cHH2_blue_smoothed,* cHH3_blue_smoothed, * cHH1_blue_orig, * cHH2_blue_orig, * cHH3_blue_orig ;
    int ib1r,  ib1c,  ib2r,  ib2c,  ib3r, ib3c;
    int ibs1r,  ibs1c,  ibs2r,  ibs2c,  ibs3r,  ibs3c;

    const char* name = "db2";
    int N = dst.rows * dst.cols;
    obj_blue_smoothed = wave_init(name);
    obj_blue_orig = wave_init(name);

    wt_blue_orig = wt2_init(obj_blue_orig, "dwt", workImg.rows, workImg.cols, J);
    wt_blue_smoothed = wt2_init(obj_blue_smoothed, "dwt", dst.rows, dst.cols, J);

    wavecoeffs_blue_orig = dwt2(wt_blue_orig, blue_orig);
    wavecoeffs_blue_smoothed = dwt2(wt_blue_smoothed, blue_smoothed);

    cHH1_blue_orig = getWT2Coeffs(wt_blue_orig, wavecoeffs_blue_orig, 1, 'D', &ib1r, &ib1c);
    cHH2_blue_orig = getWT2Coeffs(wt_blue_orig, wavecoeffs_blue_orig, 2, 'D', &ib2r, &ib2c);
    cHH3_blue_orig = getWT2Coeffs(wt_blue_orig, wavecoeffs_blue_orig, 3, 'D', &ib3r, &ib3c);

    cHH1_blue_smoothed = getWT2Coeffs(wt_blue_smoothed, wavecoeffs_blue_smoothed, 1, 'D', &ibs1r, &ibs1c);
    cHH2_blue_smoothed = getWT2Coeffs(wt_blue_smoothed, wavecoeffs_blue_smoothed, 2, 'D', &ibs2r, &ibs2c);
    cHH3_blue_smoothed = getWT2Coeffs(wt_blue_smoothed, wavecoeffs_blue_smoothed, 3, 'D', &ibs3r, &ibs3c);

    //dispWT2Coeffs(cHH1s, i1r, i1c);
    std::cout << ibs1r << " " << ibs1c << std::endl;

   medianaa(cHH1_blue_orig, cHH1_blue_smoothed, ibs1r, ibs1c);
   medianaa(cHH2_blue_orig, cHH2_blue_smoothed, ibs2r, ibs2c);
   medianaa(cHH3_blue_orig, cHH3_blue_smoothed, ibs3r, ibs3c);

   oup_blue = (double*)calloc(N, sizeof(double));
   for (int i = 0; i < dst.rows; ++i) {
       for (int k = 0; k < dst.cols; ++k) {
           oup_blue[i * dst.cols + k] = 0.0;
       }
   }
   idwt2(wt_blue_smoothed, wavecoeffs_blue_smoothed, oup_blue);

   // GREEN
   wave_object obj_green_smoothed, obj_green_orig;
   wt2_object wt_green_smoothed, wt_green_orig;

   double* wavecoeffs_green_smoothed, * wavecoeffs_green_orig, * oup_green;
   double* cHH1_green_smoothed, * cHH2_green_smoothed, * cHH3_green_smoothed, * cHH1_green_orig, * cHH2_green_orig, * cHH3_green_orig;
   int ig1r, ig1c, ig2r, ig2c, ig3r, ig3c;
   int igs1r, igs1c, igs2r, igs2c, igs3r, igs3c;

   obj_green_smoothed = wave_init(name);
   obj_green_orig = wave_init(name);

   wt_green_orig = wt2_init(obj_green_orig, "dwt", workImg.rows, workImg.cols, J);
   wt_green_smoothed = wt2_init(obj_green_smoothed, "dwt", dst.rows, dst.cols, J);

   wavecoeffs_green_orig = dwt2(wt_green_orig, green_orig);
   wavecoeffs_green_smoothed = dwt2(wt_green_smoothed, green_smoothed);

   cHH1_green_orig = getWT2Coeffs(wt_green_orig, wavecoeffs_green_orig, 1, 'D', &ig1r, &ig1c);
   cHH2_green_orig = getWT2Coeffs(wt_green_orig, wavecoeffs_green_orig, 2, 'D', &ig2r, &ig2c);
   cHH3_green_orig = getWT2Coeffs(wt_green_orig, wavecoeffs_green_orig, 3, 'D', &ig3r, &ig3c);

   cHH1_green_smoothed = getWT2Coeffs(wt_green_smoothed, wavecoeffs_green_smoothed, 1, 'D', &igs1r, &igs1c);
   cHH2_green_smoothed = getWT2Coeffs(wt_green_smoothed, wavecoeffs_green_smoothed, 2, 'D', &igs2r, &igs2c);
   cHH3_green_smoothed = getWT2Coeffs(wt_green_smoothed, wavecoeffs_green_smoothed, 3, 'D', &igs3r, &igs3c);

   medianaa(cHH1_green_orig, cHH1_green_smoothed, igs1r, igs1c);
   medianaa(cHH2_green_orig, cHH2_green_smoothed, igs2r, igs2c);
   medianaa(cHH3_green_orig, cHH3_green_smoothed, igs3r, igs3c);

   oup_green = (double*)calloc(N, sizeof(double));
   for (int i = 0; i < dst.rows; ++i) {
       for (int k = 0; k < dst.cols; ++k) {
           oup_green[i * dst.cols + k] = 0.0;
       }
   }
   idwt2(wt_green_smoothed, wavecoeffs_green_smoothed, oup_green);

    // RED
   wave_object obj_red_smoothed, obj_red_orig;
   wt2_object wt_red_smoothed, wt_red_orig;

   double* wavecoeffs_red_smoothed, * wavecoeffs_red_orig, * oup_red;
   double* cHH1_red_smoothed, * cHH2_red_smoothed, * cHH3_red_smoothed, * cHH1_red_orig, * cHH2_red_orig, * cHH3_red_orig;
   int ir1r, ir1c, ir2r, ir2c, ir3r, ir3c;
   int irs1r, irs1c, irs2r, irs2c, irs3r, irs3c;

   obj_red_smoothed = wave_init(name);
   obj_red_orig = wave_init(name);

   wt_red_orig = wt2_init(obj_red_orig, "dwt", workImg.rows, workImg.cols, J);
   wt_red_smoothed = wt2_init(obj_red_smoothed, "dwt", dst.rows, dst.cols, J);

   wavecoeffs_red_orig = dwt2(wt_red_orig, red_orig);
   wavecoeffs_red_smoothed = dwt2(wt_red_smoothed, red_smoothed);

   cHH1_red_orig = getWT2Coeffs(wt_red_orig, wavecoeffs_red_orig, 1, 'D', &ir1r, &ir1c);
   cHH2_red_orig = getWT2Coeffs(wt_red_orig, wavecoeffs_red_orig, 2, 'D', &ir2r, &ir2c);
   cHH3_red_orig = getWT2Coeffs(wt_red_orig, wavecoeffs_red_orig, 3, 'D', &ir3r, &ir3c);

   cHH1_red_smoothed = getWT2Coeffs(wt_red_smoothed, wavecoeffs_red_smoothed, 1, 'D', &irs1r, &irs1c);
   cHH2_red_smoothed = getWT2Coeffs(wt_red_smoothed, wavecoeffs_red_smoothed, 2, 'D', &irs2r, &irs2c);
   cHH3_red_smoothed = getWT2Coeffs(wt_red_smoothed, wavecoeffs_red_smoothed, 3, 'D', &irs3r, &irs3c);

   medianaa(cHH1_red_orig, cHH1_red_smoothed, irs1r, irs1c);
   medianaa(cHH2_red_orig, cHH2_red_smoothed, irs2r, irs2c);
   medianaa(cHH3_red_orig, cHH3_red_smoothed, irs3r, irs3c);

   oup_red = (double*)calloc(N, sizeof(double));
   for (int i = 0; i < dst.rows; ++i) {
       for (int k = 0; k < dst.cols; ++k) {
           oup_red[i * dst.cols + k] = 0.0;
       }
   }
   idwt2(wt_green_smoothed, wavecoeffs_green_smoothed, oup_green);
   // End of wavelet

   cv::Mat final_blue(dst.rows, dst.cols, CV_64F, oup_blue);
   cv::Mat final_green(dst.rows, dst.cols, CV_64F, oup_green);
   cv::Mat final_red(dst.rows, dst.cols, CV_64F, oup_red);

   cv::Mat convertedMat_blue, convertedMat_green, convertedMat_red;
   final_blue.convertTo(convertedMat_blue, CV_8U);
   final_green.convertTo(convertedMat_green, CV_8U);
   final_red.convertTo(convertedMat_red, CV_8U);

   cv::Mat waveChannels[3] = { convertedMat_blue, convertedMat_green, convertedMat_red };


   cv::Mat final_colors;
   cv::merge(waveChannels, 3, final_colors);

   cv::imshow("FINALLY", final_colors);


    cv::waitKey();
    cv::destroyAllWindows();


    // free blue 
    wave_free(obj_blue_smoothed);
    wt2_free(wt_blue_smoothed);
    wave_free(obj_blue_orig);
    wt2_free(wt_blue_orig);
    free(wavecoeffs_blue_orig);
    free(wavecoeffs_blue_smoothed);
    free(oup_blue);

    // free green
    wave_free(obj_green_smoothed);
    wt2_free(wt_green_smoothed);
    wave_free(obj_green_orig);
    wt2_free(wt_green_orig);
    free(wavecoeffs_green_orig);
    free(wavecoeffs_green_smoothed);
    free(oup_green);

    // free red
    wave_free(obj_red_smoothed);
    wt2_free(wt_red_smoothed);
    wave_free(obj_red_orig);
    wt2_free(wt_red_orig);
    free(wavecoeffs_red_orig);
    free(wavecoeffs_red_smoothed);
    free(oup_red);
    return 0;
}
