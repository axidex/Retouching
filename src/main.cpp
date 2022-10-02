/**
 * @file BinarySkinMask.cpp
 * @brief An example of how to create a binary skin mask from face landmark locations.
 *
 */
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/ml.hpp>

#include <ranges>
#include <tinysplinecxx.h>
#include <iostream>

 /// @brief Command line keys for command line parsing
static constexpr auto cmdKeys =
"{help h usage ?   |       | print this message            }"
"{@image           |<none> | input image                   }"
"{@landmark_model  |<none> | face landmark detection model }"
"{images_dir       |       | search path for images        }"
"{models_dir       |       | search path for models        }";

static void cvHaarWavelet(cv::Mat& src, cv::Mat& dst, int NIter)
{
    float c, dh, dv, dd;
    assert(src.type() == CV_32FC1);
    assert(dst.type() == CV_32FC1);
    int width = src.cols;
    int height = src.rows;
    for (int k = 0; k < NIter; k++)
    {
        for (int y = 0; y < (height >> (k + 1)); y++)
        {
            for (int x = 0; x < (width >> (k + 1)); x++)
            {
                c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y, x) = c;

                dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y, x + (width >> (k + 1))) = dh;

                dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y + (height >> (k + 1)), x) = dv;

                dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
            }
        }
        dst.copyTo(src);
    }
}



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



/// @brief Bianry skin mask example
///
/// Usage: BinarySkinMask.exe [params] image landmark_model
int main(int argc, char** argv) {
    // Handle command line arguments
    cv::CommandLineParser parser(argc, argv, cmdKeys);
    parser.about("Bianry skin mask example");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (parser.has("images_dir"))
        cv::samples::addSamplesDataSearchPath(parser.get<cv::String>("images_dir"));
    if (parser.has("models_dir"))
        cv::samples::addSamplesDataSearchPath(parser.get<cv::String>("models_dir"));

    // Load image
    const auto imgArg = parser.get<cv::String>("@image");
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }
    // Set `required=false` to prevent `findFile` from throwing an exception.
    // Instead, we check whether the image is valid via the `empty` method.
    const auto inputImg =
        cv::imread(cv::samples::findFile(imgArg, /*required=*/false, /*silentMode=*/true));
    if (inputImg.empty()) {
        std::cout << "Could not open or find the image: " << imgArg << "\n"
            << "The image should be located in `images_dir`.\n";
        parser.printMessage();
        return -1;
    }
    // Make a copy for drawing landmarks
    cv::Mat landmarkImg = inputImg.clone();
    // Make a copy for drawing binary mask
    cv::Mat maskImg = cv::Mat::zeros(inputImg.size(), CV_8UC1);

    // Load dlib's face landmark detection model
    const auto landmarkModelArg = parser.get<cv::String>("@landmark_model");
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }
    auto landmarkModelPath = cv::samples::findFile(landmarkModelArg, /*required=*/false);
    if (landmarkModelPath.empty()) {
        std::cout << "Could not find the landmark model file: " << landmarkModelArg << "\n"
            << "The model should be located in `models_dir`.\n";
        parser.printMessage();
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
    int dx;
    double fc;
    std::cin >> dx;
    std::cin >> fc;
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
  
    //cv::imshow("ma", ma);
   // cv::imwrite("final.jpg", tmp2);

    /*
    // Fit image to the screen and show image
    cv::namedWindow(landmarkWin, cv::WINDOW_NORMAL);
    cv::setWindowProperty(landmarkWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    const auto [x, y, resW, resH] = cv::getWindowImageRect(landmarkWin);
    const auto [imgW, imgH] = landmarkImg.size();
    const auto scaleFactor = 40;
    const auto scaledW = scaleFactor * resW / 100;
    const auto scaledH = scaleFactor * imgH * resW / (imgW * 100);
    cv::resizeWindow(landmarkWin, scaledW, scaledH);*/
    // Show overlay
    // cv::Mat maskTmp[3] = {maskImg, maskImg, maskImg};
    // cv::Mat mask;
    // cv::merge(maskTmp, 3, mask);
    // cv::bitwise_and(mask, landmarkImg, landmarkImg);
    /* cv::imshow(landmarkWin, landmarkImg);

    // Show binary skin mask
    cv::namedWindow(skinMaskWin, cv::WINDOW_NORMAL);
    cv::setWindowProperty(skinMaskWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::resizeWindow(skinMaskWin, scaledW, scaledH);
    cv::moveWindow(skinMaskWin, scaledW, 0);
    cv::imshow(skinMaskWin, maskImg);
    */
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}
