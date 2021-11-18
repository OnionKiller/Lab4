// Lab4.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <format>
#include <bitset>

struct Marker
{
	std::vector<cv::Point2f> points;
	/***
	* FU OpenCV, and your kinky interfaces 
	*/
	std::vector<std::vector<cv::Point>> contr() {
		std::vector<cv::Point> r_{};
		for (auto I : points)
			r_.push_back(I);
		return { r_ };
	}
};

void performThreshold(const cv::Mat& grayscale, cv::Mat& thresholdImg)
{
	cv::adaptiveThreshold(
		grayscale,
		thresholdImg,
		255,
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY_INV,
		7,
		7
	);
}

void findContours(const cv::Mat& thresholdImg,
	std::vector<std::vector<cv::Point>>& contours,
	int minContourPointsAllowed)
{
	std::vector< std::vector<cv::Point> > allContours;
	cv::findContours(thresholdImg, allContours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	contours.clear();
	for (size_t i = 0; i < allContours.size(); i++)
	{
		int contourSize = allContours[i].size();
		if (contourSize > minContourPointsAllowed)
		{
			contours.push_back(allContours[i]);
		}
	}
}

float perimeter(const Marker& m)
{
	float perim = 0;
	for (int i = 0; i < m.points.size() - 1; ++i)
	{
		cv::Point p1 = m.points[i];
		cv::Point p2 = m.points[i + 1];
		cv::Point edge = p2 - p1;
		perim += sqrt(edge.dot(edge));
	}
	return perim;
}

void findMarkerQuads(std::vector<std::vector<cv::Point>>& contours,
	std::vector<Marker>& markers,
	float minContourLengthAllowed)
{
	std::vector<cv::Point> approxCurve;

	for (size_t i = 0; i < contours.size(); i++)
	{
		// Approximate to a polygon
		double eps = contours[i].size() * 0.05;
		cv::approxPolyDP(contours[i], approxCurve, eps, true);
		// We interested only in polygons that contains only four points
		if (approxCurve.size() != 4)
			continue;
		// And they have to be convex
		if (!cv::isContourConvex(approxCurve))
			continue;
		// Ensure that the distance between consecutive points is large enough
		float minDist = std::numeric_limits<float>::max();
		for (int i = 0; i < 4; i++)
		{
			cv::Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredSideLength = side.dot(side);
			minDist = std::min(minDist, squaredSideLength);
		}
		// Check that distance is not very small
		if (minDist < minContourLengthAllowed)
			continue;
		// All tests are passed. Save marker candidate:
		Marker m;
		for (int i = 0; i < 4; i++)
			m.points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));
		// Sort the points in anti-clockwise order
		// Trace a line between the first and second point.
		// If the third point is at the right side, then the pointsare anticlockwise
		cv::Point v1 = m.points[1] - m.points[0];
		cv::Point v2 = m.points[2] - m.points[0];
		double o = (v1.x * v2.y) - (v1.y * v2.x);
		if (o < 0.0) //if the third point is in the left side, then sort in anti - clockwise order
			std::swap(m.points[1], m.points[3]);
		markers.push_back(m);
	}
}


void removeMarkerDuplicates(std::vector<Marker>& markersIn,
	std::vector<Marker>& markersOut)
{
	// Remove these elements which corners are too close to each other.
	// First detect candidates for removal:
	std::vector< std::pair<int, int> > tooNearCandidates;
	for (size_t i = 0; i < markersIn.size(); i++)
	{
		const Marker& m1 = markersIn[i];
		//calculate the average distance of each corner to the nearest corner of the other marker candidate
		for (size_t j = i + 1; j < markersIn.size(); j++)
		{
			const Marker& m2 = markersIn[j];
			float distSquared = 0;
			for (int c = 0; c < 4; c++)
			{
				cv::Point v = m1.points[c] - m2.points[c];
				distSquared += v.dot(v);
			}
			distSquared /= 4;
			if (distSquared < 100)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}
	// Mark for removal the element of the pair with smaller perimeter
	std::vector<bool> removalMask(markersIn.size(), false);
	for (size_t i = 0; i < tooNearCandidates.size(); i++)
	{
		float p1 = perimeter(markersIn[tooNearCandidates[i].first]);
		float p2 = perimeter(markersIn[tooNearCandidates[i].second]);
		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;
		removalMask[removalIndex] = true;
	}
	// Return candidates
	markersOut.clear();
	for (size_t i = 0; i < markersIn.size(); i++)
	{
		if (!removalMask[i])
			markersOut.push_back(markersIn[i]);
	}
}

void removePerspective(const cv::Mat& image, std::vector<Marker>& markers, std::vector<cv::Mat>& canonicalMarkers)
{
	cv::Size markerSize(100, 100);
	std::vector<cv::Point2f> origPoints;
	origPoints.push_back(cv::Point2f(0, 0));
	origPoints.push_back(cv::Point2f(markerSize.width - 1, 0));
	origPoints.push_back(cv::Point2f(markerSize.width - 1, markerSize.height - 1));
	origPoints.push_back(cv::Point2f(0, markerSize.height - 1));

	cv::Mat canonicalMarker;

	for (size_t iMarker = 0; iMarker < markers.size(); ++iMarker)
	{
		Marker& marker = markers[iMarker];
		// getPerspectiveTransform requires points with floating point coordinates
		std::vector<cv::Point2f> points;
		for (int i = 0; i < marker.points.size(); ++i)
		{
			points.push_back(marker.points[i]);
		}

		// Find the perspective transfomation that brings current marker to rectangular form
		cv::Mat M = cv::getPerspectiveTransform(points, origPoints);
		// Transform image to get a canonical marker image
		cv::warpPerspective(image, canonicalMarker, M, markerSize);

		canonicalMarkers.push_back(canonicalMarker);
	}
}

int get_id(cv::Mat canonoical_marker,cv::Rect dataROI,cv::Size s) {
	CV_Assert(dataROI.width == dataROI.height);
	cv::Mat c_marker, data;
	cv::resize(canonoical_marker, c_marker, s);
	data = c_marker(dataROI);
	int iter = dataROI.width / 5;
	std::array<std::bitset<5>, 5>code;
	size_t id = 0;
	for (int i = 0; i < 5; i++)
	for (int j = 0; j < 5; j++)
	{
		cv::Rect ROI{ iter * j,iter * i,iter,iter };
		if (cv::sum(data(ROI))[0] < 127. * ROI.area())
			code[i][j] = 0;
		else
			code[i][j] = 1;
		if (j == 1 || j == 3)
		{
			id = id << 1;
			id += code[i][j];
		}
	}
	return id;
}



int main(int argc, char* argv[])
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
		cv::Mat gray,bw;
		cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
		performThreshold(gray, bw);

		std::vector< std::vector<cv::Point> > contours;
		findContours(bw, contours, 100);

		cv::Mat contourImage = cv::Mat::zeros(bw.rows, bw.cols, CV_8UC3);
		std::vector<Marker> quads{};
		findMarkerQuads(contours, quads, 10);
		auto nd_quads = quads;
		removeMarkerDuplicates(quads, nd_quads);

		for (auto q : nd_quads)
		{
			drawContours(contourImage,q.contr() , -1, cv::Scalar(255, 255, 255));
		}
		cv::imshow("PreviewWindow", contourImage);


		std::vector<cv::Mat> markers{};
		removePerspective(gray, nd_quads, markers);

		for (auto &I : markers)
		{
			cv::threshold(I, I, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
			cv::resize(I, I, {}, 2, 2);
		}

		size_t i = 0;
		for (auto I : markers)
		{
			cv::imshow("marker"+std::to_string(i++), I);
			auto id = get_id(I, { 30,30,140,140 }, { 200,200 });
			std::cout << std::format("Marker {} id is: {}", i, id) << std::endl;
		}

		//cv::imshow("PreviewWindow", gray);

		char key = static_cast<char>(cv::waitKey());
		if (key == 27) { printf("exit"); break; }
	}

}