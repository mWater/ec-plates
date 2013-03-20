#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "Circle.h"
#include "CircleFinder.h"


using namespace cv;

static void timeit(const char *name) {
}

/*
 * Finds the rectangle which fits around the Petri dish circle that has been detected
 */
Rect findPetriRect(Mat img)
{
	Vec3f circ = findPetriDish(img);
	return Rect(circ[0]-circ[2], circ[1]-circ[2], circ[2]*2, circ[2]*2);
}

// maximum size of the image to process.  If larger, will be scaled
static int maxSize = 1024;

// Finds edges in the image
static Mat findEdges(Mat img)
{
	static int cannyParam1 = 30;

	// Smooth image
	GaussianBlur(img, img, Size(9,9), 1, 1);

	// Find edges 
	Mat edges;
	Canny(img, edges, cannyParam1, cannyParam1/2);

	return edges;
}

/*
 * Finds the most likely centerpoint of circles that are present in a set of contours.
 * It does this by randomly sampling sets of three points from each contour.
 * With these three points, it determines the circle that passes through these three points.
 * The circle center is then incremented in another zeroed matrix.
 * The point that has the most number of increments is the most likely center.
 * This is done instead of HoughCircles as the Petri dish has multiple near-concentric
 * circles, which confuses the HoughCircles algorithm
 */
void findBestCenter(Size imgSize, vector<vector<Point> > contours, double& maxVal, Point& maxLoc, bool debug)
{
	static int iterations = 10000;
	static int minDistStartEnd = 40;
	static double minRadius = maxSize / 10;

	// Create array to total possible centers in
	Mat centers(imgSize, CV_32F, Scalar(0.0));

	if (debug)
		timeit("finding centers...");

	// Start finding circles
	for (int iter=0;iter<iterations;iter++)
	{
		// Select random contour
		int ctr = rand() % contours.size();

		// Pick random start point
		Point2d start = contours[ctr][rand() % contours[ctr].size()];

		// Pick random end point
		Point2d end = contours[ctr][rand() % contours[ctr].size()];

		// If not far enough apart, skip
		double dist = norm(start-end);
		if (dist<minDistStartEnd)
			continue;

		// Pick random third point
		Point2d third = contours[ctr][rand() % contours[ctr].size()];
		if (norm(third-start)<minDistStartEnd)
			continue;
		if (norm(third-end)<minDistStartEnd)
			continue;

		// Find circle of best fit
		Circle circ(start, end, third);
		if (circ.GetRadius() < minRadius)
			continue;

		// Put center on image to total where centers are
		if (circ.GetCenter().inside(Rect(0,0,imgSize.width, imgSize.height)))
			centers.at<float>(circ.GetCenter().y, circ.GetCenter().x)+=1.0;
	}

	if (debug)
		timeit("centers");

	// Smooth centers
	GaussianBlur(centers, centers, Size(9, 9), 1, 1);

	// Find best center
	minMaxLoc(centers, NULL, &maxVal, NULL, &maxLoc);

	if (debug)
	{
		// Histogram adjust
		centers = centers * (1/maxVal);

		// Draw ref point
		line(centers, maxLoc-Point(0,20), maxLoc-Point(0,40), Scalar(1));
		line(centers, maxLoc-Point(20,0), maxLoc-Point(40,0), Scalar(1));

		// Draw contours
		drawContours(centers, contours, -1, Scalar(0.5));
		imshow("centers", centers);
	}
}

/*
 * Finds the circle of the Petri dish within an image.
 * It does so by iteratively finding the strongest circle, eliminating all contours outside of it
 * and then looking for any further circles within it, down to a minimum radius based on the
 * original circle.
 */
Vec3f findPetriDish(Mat img)
{
	bool debug = false;							// True to display progress images
	static int minContourSize = 120;			// Minimum size in pixels of a contour to be considered
	static int minDistStartEnd = 30;			// Minimum distance between a start and end point of any two of the three points
	static double minRadius = maxSize / 10;		// Minimum radius of the circle
	int minContourPoints = 15;					// Minimum number of contour points in a contour
	double minCenterVal = 1;					// Minimum accumulated center value

	Point center(0,0);
	double radius = 0;

	// Convert to scaled grayscale image
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	// Scale to 1024 size
	double scaleby = max(gray.rows, gray.cols)*1.0/maxSize;
	Mat resized;
	resize(gray, resized, Size(), 1.0/scaleby, 1.0/scaleby, INTER_CUBIC);
	gray = resized;

	if (debug)
		timeit(NULL);

	// Find edges in the image
	Mat edges = findEdges(gray);

	if (debug)
		timeit("resize and edges");

	// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (debug)
		timeit("contours");

	bool firstIter = true;

	do {
		// Remove small contours and contours with few points
		vector<vector<Point> > contours2;
		for (int c=0;c<contours.size();c++) {
			Rect r = boundingRect(contours[c]);

			if (r.width > minContourSize || r.height > minContourSize) {
				if (contours[c].size() >= minContourPoints)
					contours2.push_back(contours[c]);
			}
		}
		contours=contours2;

		if (contours.size() == 0)
			break;

		// Find best center
		double maxVal;
		Point maxLoc;
		findBestCenter(edges.size(), contours, maxVal, maxLoc, debug);

		// If center is in sufficiently strong, exit
		if (maxVal < minCenterVal)
			break;

		// Ensure that points on at least 3 quadrants are present to prevent small arcs from causing errors
		// Arcs that do not include at least three quadrants are likely to give erroneous centerpoints
		bool quadrants[] = { false, false, false, false };
		for (int c=0;c<contours.size();c++) {
			for (int i=0;i<contours[c].size();i++) {
				Point p = contours[c][i] - maxLoc;
				if (p.x > 0)
					quadrants[p.y > 0 ? 1 : 0] = true;
				else
					quadrants[p.y > 0 ? 3 : 2] = true;
			}
		}
		int totalQuadrants = 0;
		for (int i=0;i<4;i++) {
			if (quadrants[i])
				totalQuadrants++;
		}
		if (totalQuadrants<3)
			break;

		// Find distance for all contour points from center
		Mat dist(gray.rows + gray.cols, 1, CV_32F, Scalar(0));
		for (int c=0;c<contours.size();c++) {
			for (int i=0;i<contours[c].size();i++) {
				dist.at<float>(norm(contours[c][i]-maxLoc))+=1;
			}
		}

		// Find best radius
		GaussianBlur(dist, dist, Size(1, 3), 0, 1);
		double bestRadVal;
		int bestRadIndex;
		minMaxIdx(dist, NULL, &bestRadVal, NULL, &bestRadIndex);

		center = maxLoc;
		radius = bestRadIndex - 1;		// Move inside points

		// Remove any contour points not well inside circle to get ready to look again
		contours2.clear();
		for (int c=0;c<contours.size();c++)
		{
			vector<Point> ctr;
			for (int i=0;i<contours[c].size();i++)
			{
				if (norm(contours[c][i]-maxLoc) <= bestRadIndex - 4)
					ctr.push_back(contours[c][i]);
			}
			if (ctr.size() > 0)
				contours2.push_back(ctr);
		}
		contours=contours2;

		// If first iteration
		if (firstIter)
		{
			// Prevent too small circles from being found
			minRadius = radius * 0.80;
		}

		if (debug)
		{
			timeit("circle found");

			Mat img2 = img.clone();

			circle(img2, maxLoc * scaleby, bestRadIndex * scaleby, Scalar(0,255,0), 2);

			namedWindow("circ", CV_GUI_EXPANDED);
			imshow("circ", img2);

			waitKey(0);
		}

		firstIter = false;
	} while (true);

	// Move inside outer edge to avoid edge effects
	radius *= 0.975;

	// Return circle
	return Vec3f(center.x * scaleby, center.y * scaleby, radius * scaleby);
}

/*
 * Measure the performance of circle finding given a reference image
 * Ref image should be 100% green for the correct area. Ref image should
 * be 100% red for the fringes that are still within the outer limits of the Petri dish
 */
bool testCirclePerformance(Vec3f circ, Mat refImg) 
{
	// Get red region
	vector<Mat> bgr;
	split(refImg, bgr);

	Mat red, nogreen, green, nored;
	threshold(bgr[2], red, 254, 255, THRESH_BINARY);
	threshold(bgr[1], nogreen, 0, 255, THRESH_BINARY_INV);
	red = red & nogreen;

	threshold(bgr[1], green, 254, 255, THRESH_BINARY);
	threshold(bgr[2], nored, 0, 255, THRESH_BINARY_INV);
	green = green & nored;

	// Dilate green to close small gaps
	dilate(green, green, Mat());

	// Create image with circle drawn
	Mat circleImg(refImg.size(), CV_8U, Scalar(0));
	circle(circleImg, Point2f(cvRound(circ[0]), cvRound(circ[1])), cvRound(circ[2]), 
		Scalar(255), CV_FILLED);

	Mat bad = 255-(red+green);
	long greenIn = sum(green & circleImg)[0];
	long greenTot = sum(green)[0];
	long redIn = sum(red & circleImg)[0];
	long redTot = sum(red)[0];
	long badIn = sum(bad & circleImg)[0];
	printf("%5.3f caught %5.3f bad %5.3f fringe\n", (greenIn * 100.0)/greenTot,
		(badIn * 100.0) / greenTot, (redIn * 100.0)/redTot);

	return greenIn > greenTot && badIn == 0;
}
