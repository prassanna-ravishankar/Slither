#pragma once

// OpenCV Compatibility Header
// Ensures compatibility between OpenCV 3.x and 4.x versions

#include <opencv2/core/version.hpp>

// In OpenCV 4.x, some constants were moved to cv:: namespace
// This provides backward compatibility
#if CV_VERSION_MAJOR < 4
    // OpenCV 3.x - constants are in global namespace
#else
    // OpenCV 4.x - import commonly used constants to global namespace for compatibility
    using cv::CV_32FC1;
    using cv::CV_64FC1;
    using cv::CV_8UC1;
    using cv::CV_8UC3;
#endif

// ML module namespace alias for consistency
namespace cvml = cv::ml;