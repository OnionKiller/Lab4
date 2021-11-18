// Lab4.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, std::string argv[])
{
	cv::VideoCapture capture;
	cv::Mat frame;
	bool success = true;
	if (argc < 2) {
		// webcam with index 0
		success = capture.open(0);
	}
	else {
		// video file
		success = capture.open(argv[1]);
	}
	if (!success) {
		std::cerr << "Unable to open video capture" << std::endl;
		return 0;
	}
	while (capture.read(frame))
	{
		cv::Mat resized;
		cv::resize(frame, resized, cv::Size(640, 480));
		cv::Mat gray;
		cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
		cv::imshow("PreviewWindow", gray);

		char key = static_cast<char>(cv::waitKey(1));
		if (key == 27) { printf("exit"); break; }
	}

}