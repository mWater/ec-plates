#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "Circle.h"
#include "CircleFinder.h"


using namespace cv;
static void timeit(const char *name) {
}


Rect findPetriRect(Mat img)
{
	Vec3f circ = findPetriDish(img);
	return Rect(circ[0]-circ[2], circ[1]-circ[2], circ[2]*2, circ[2]*2);
}
static int maxSize = 1024;

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
		//int ctr = contourIndex[rand() % contourIndex.size()];
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


Vec3f findPetriDish(Mat img)
{
	bool debug = false;
	static int minContourSize = 120;
	static int minDistStartEnd = 40;
	static double minRadius = maxSize / 10;

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

	Mat edges = findEdges(gray);

	if (debug)
		timeit("resize and edges");

	// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (debug)
		timeit("contours");

	// Remove small contours
	vector<vector<Point> > contours2;
	for (int c=0;c<contours.size();c++) {
		Rect r = boundingRect(contours[c]);

		if (r.width > minContourSize || r.height > minContourSize)
			contours2.push_back(contours[c]);
	}
	contours=contours2;

	// Find best center
	double maxVal;
	Point maxLoc;
	findBestCenter(edges.size(), contours, maxVal, maxLoc, debug);

	// Find distance for all contour points from center
	Mat dist(gray.rows + gray.cols, 1, CV_32F, Scalar(0));
	for (int c=0;c<contours.size();c++) {
		for (int i=0;i<contours[c].size();i++) {
			dist.at<float>(norm(contours[c][i]-maxLoc))+=1;
		}
	}

	// Find integral of distances, as we want to find good range from 85% to 100% of radius
	// This represents the plastic disk around outside of lid
	Mat distInt;
	integral(dist, distInt);

	// Find best match for 88% inner circle
	int bestRadIndex = -1;
	double bestRadSum = 0;
	for (int r=minRadius;r<distInt.rows;r++) {
		double val = distInt.at<double>(r, 1) - distInt.at<double>(r*0.88, 1);
		val/=r*r;
		if (val > bestRadSum || bestRadIndex == -1) {
			bestRadSum = val;
			bestRadIndex = r;
		}
	}

	if (debug)
	{
		timeit("circle found");

		Mat img2 = img.clone();

		circle(img2, maxLoc * scaleby, bestRadIndex * scaleby, Scalar(0,255,0), 2);
		circle(img2, maxLoc * scaleby, bestRadIndex * 0.88 *  scaleby, Scalar(0,255,0), 2);
		circle(img2, maxLoc * scaleby, bestRadIndex * 0.94 *  scaleby, Scalar(0,0,255), 2);

		namedWindow("circ", CV_GUI_EXPANDED);
		imshow("circ", img2);

		namedWindow("edges", CV_GUI_EXPANDED);
		imshow("edges", edges);
		waitKey(0);
	}
	
	// Return circle at 94% radius of outside ring to allow for error in detection
	return Vec3f(maxLoc.x * scaleby, maxLoc.y * scaleby, bestRadIndex * scaleby * 0.94);
}

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


//#include <unsupported/Eigen/NonLinearOptimization>
//
//static struct CircCand {
//	Point2d center;
//	double radius;
//	double goodness;
//	double relatedgoodness;
//};
//
//Vec3f findPetriDish(Mat img) 
//{
//	static int cannyParam1 = 40;
//	static int maxSize = 1024;
//	static int minContourSize = 80;
//	static int minDistStartEnd = 50;
//	static double minCircularness = 0.1;
//	static double minRadius = maxSize / 10;
//
//	// Convert to scaled grayscale image
//	Mat gray;
//	cvtColor(img, gray, CV_BGR2GRAY);
//
//	// Scale to 1024 size
//	double scaleby = max(gray.rows, gray.cols)*1.0/maxSize;
//	Mat resized;
//	resize(gray, resized, Size(), 1.0/scaleby, 1.0/scaleby, INTER_CUBIC); 
//	gray = resized;
//
//	// Find edges 
//	Mat edges;
//	Canny(gray, edges, cannyParam1, cannyParam1/2);
//
//	// Find contours
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	findContours(edges, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//
//	// Remove small contours
//	vector<vector<Point> > contours2;
//	for (int c=0;c<contours.size();c++) {
//		Rect r = boundingRect(contours[c]);
//
//		if (r.width > minContourSize || r.height > minContourSize)
//			contours2.push_back(contours[c]);
//	}
//	contours=contours2;
//
//	// Find candidate circles
//	vector<CircCand> cands;
//
//	// Start finding circles
//	for (int iter=0;iter<2000;iter++) 
//	{
//		// Select random contour
//		int ctr = rand() % contours.size();
//		
//		// Pick random start point
//		Point2d start = contours[ctr][rand() % contours[ctr].size()];
//
//		// Pick random end point
//		Point2d end = contours[ctr][rand() % contours[ctr].size()];
//
//		// If not far enough apart, skip
//		double dist = norm(start-end);
//		if (dist<minDistStartEnd) 
//			continue;
//
//		// Find best third point between (closest to line perpendicular to line between two points)
//		// TODO doc and extract const
//		Point2d startend = (end-start)*(1.0/dist);
//		int bestThirdIdx = -1;
//		double bestError = 0;
//		for (int i=0;i<contours[ctr].size();i++) 
//		{
//			Point2d fromstart = ((Point2d)contours[ctr][i])-start;
//			double error = fabs(fromstart.ddot(startend) - dist/2);
//			if (bestThirdIdx == -1 || error < bestError) {
//				bestError = error;
//				bestThirdIdx = i;
//			}
//		}
//		if (bestThirdIdx == -1 || bestError > 2)
//			continue;
//
//		// Get circularness
//		Point2d third = contours[ctr][bestThirdIdx];
//		Point2d dev = third - (end+start)*0.5;
//		if (norm(dev)/dist < minCircularness)
//			continue;
//
//		// Find circle of best fit
//		Circle circ(start, end, third);
//		if (circ.GetRadius() < minRadius)
//			continue;
//		
//		// Check that at least n points near circle
//		int good = 0;
//		for (int c=0;c<contours.size();c++) {
//			for (int i=0;i<contours[c].size();i++) 
//			{
//				Point2d rel = ((Point2d)contours[c][i])-circ.GetCenter();
//				double err = norm(rel) - circ.GetRadius();
//				if (fabs(err) < 3) // ###
//					good++;
//			}
//		}
//		//if (good < 50)
//		//	continue;
//
//		CircCand cand;
//		cand.center = circ.GetCenter();
//		cand.radius = circ.GetRadius();
//		cand.goodness = good;
//		cands.push_back(cand);
//	}
//
//	// Sum candidates with same center
//	for (int i=0;i<cands.size();i++) {
//		cands[i].relatedgoodness = cands[i].goodness;
//		for (int k=0;k<cands.size();k++) {
//			if (k==i)
//				continue;
//			if (norm(cands[i].center - cands[k].center) <= 2)
//				cands[i].relatedgoodness+=cands[k].goodness;
//		}
//	}
//	
//	// Find best circles
//	double bestGoodness = 0;
//	int bestIdx = -1;
//	for (int i=0;i<cands.size();i++)
//		if (cands[i].relatedgoodness>bestGoodness) {
//			bestGoodness = cands[i].relatedgoodness;
//			bestIdx = i;
//		}
//
//	Mat img2 = img.clone();
//
//	for (int i=0;i<cands.size();i++) {
//		if (cands[i].relatedgoodness < bestGoodness * 0.2)
//			continue;
//
//		// Keep concentric
//		if (norm(cands[i].center - cands[bestIdx].center) > 2)
//			continue;
//
//		// draw the circle center
//		circle( img2, cands[i].center*scaleby, 2, Scalar(0,255,0) , -1, 8, 0 );
//		// draw the circle outline
//		circle( img2, cands[i].center*scaleby, cands[i].radius*scaleby, Scalar(i==bestIdx ? 255: 0,0,255*cands[i].relatedgoodness/bestGoodness), 1, 8, 0 );
//	}
//
//	namedWindow("circ", CV_GUI_EXPANDED);
//	imshow("circ", img2);
//	waitKey(0);
//
//	return Vec3f(0,0,0);
//}
//
//
//double huber(double x, double a) 
//{
//	double signx = x > 0 ? 1 : -1;
//	double ax = fabs(x);
//	if (ax<a)
//		return x;
//	return signx * sqrt(2*a*ax - a*a);
//}
//
//double huberd1(double x, double a) 
//{	
//	double signx = x > 0 ? 1 : -1;
//	double ax = fabs(x);
//	if (ax<a)
//		return 1;
//	return a/sqrt(-a * (a-2*ax));
//}
//
//struct CircleFunctor
//{
//	double hubera;
//
//	CircleFunctor(vector<Point> points) : points(points)
//	{
//		hubera=5;
//	}
//
//	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
//	{
//		for (int i=0;i<points.size();i++)
//			fvec(i)=huber((x(0) - points[i].x) * (x(0) - points[i].x) + (x(1) - points[i].y) * (x(1) - points[i].y) - x(2) * x(2), hubera);
//		return 0;
//	}
//
//	int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
//	{
//		Eigen::VectorXd fvec(values());
//		for (int i=0;i<points.size();i++)
//			fvec(i)=(x(0) - points[i].x) * (x(0) - points[i].x) + (x(1) - points[i].y) * (x(1) - points[i].y) - x(2) * x(2);
//
//		for (int i=0;i<points.size();i++) {
//			fjac(i,0)=2*(x(0)-points[i].x)*huberd1(fvec(i),hubera);
//			fjac(i,1)=2*(x(1)-points[i].y)*huberd1(fvec(i),hubera);
//			fjac(i,2)=-2*x(2)*huberd1(fvec(i),hubera);
//		}
//		return 0;
//	};
//	//int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
//	//{
//	//	for (int i=0;i<points.size();i++)
//	//		fvec(i)=(x(0) - points[i].x) * (x(0) - points[i].x) + (x(1) - points[i].y) * (x(1) - points[i].y) - x(2) * x(2);
//	//	return 0;
//	//}
//
//	//int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
//	//{
//	//	for (int i=0;i<points.size();i++) {
//	//		fjac(i,0)=2*(x(0)-points[i].x);
//	//		fjac(i,1)=2*(x(1)-points[i].y);
//	//		fjac(i,2)=-2*x(2);
//	//	}
//	//	return 0;
//	//};
//
//	int inputs() const { return 3; };// inputs is the dimension of x.
//	int values() const { return points.size(); }; // "values" is the number of f_i and 
//
//	enum {
//	    InputsAtCompileTime = 0,
//	    ValuesAtCompileTime = 0
//	};
//	typedef double Scalar;
//	typedef Eigen::VectorXd InputType;
//	typedef Eigen::VectorXd ValueType;
//	typedef Eigen::MatrixXd JacobianType;
//
//	vector<Point> points;
//};
//
//
//
//Vec3f fitCircleToPoints(vector<Point> points, Vec3f initialGuess) 
//{
//	Eigen::VectorXd x(3);
//	for (int i=0;i<3;i++)
//		x(i)=initialGuess[i];
//
//	CircleFunctor functor(points);
//	Eigen::LevenbergMarquardt<CircleFunctor, double> lm(functor);
//	Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
//
//	return Vec3f(x(0), x(1), x(2));
//}
//
//
//Vec3f findPetriDish2(Mat img) 
//{
//	static int cannyParam1 = 40;
//	static int maxSize = 1024;
//	static int minContourSize = 80;
//	static double minRadius = maxSize / 10;
//
//	// Convert to scaled grayscale image
//	Mat gray;
//	cvtColor(img, gray, CV_BGR2GRAY);
//
//	// Scale to 1024 size
//	double scaleby = max(gray.rows, gray.cols)*1.0/maxSize;
//	Mat resized;
//	resize(gray, resized, Size(), 1.0/scaleby, 1.0/scaleby, INTER_CUBIC); 
//	gray = resized;
//
//	timeit(NULL);
//
//	// Find edges 
//	Mat edges;
//	Canny(gray, edges, cannyParam1, cannyParam1/2);
//
//	// Find contours
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	findContours(edges, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//
//	// Remove small contours
//	vector<vector<Point> > contours2;
//	for (int c=0;c<contours.size();c++) {
//		Rect r = boundingRect(contours[c]);
//
//		if (r.width > minContourSize || r.height > minContourSize)
//			contours2.push_back(contours[c]);
//	}
//	contours=contours2;
//
//	// Find candidate circles
//	vector<CircCand> cands;
//	for (int c=0;c<contours.size();c++) {
//		Point2f center;
//		float radius;
//
//		// Get initial estimate
//		minEnclosingCircle(contours[c], center, radius);
//		Vec3f circ = fitCircleToPoints(contours[c], Vec3f(center.x, center.y, radius));
//		CircCand cand;
//		cand.center = Point2f(circ[0], circ[1]);
//		cand.radius = circ[2];
//		cands.push_back(cand);
//	}
//
//	timeit("GOT IT");
//
//	Mat img2 = img.clone();
//
//	for (int i=0;i<cands.size();i++) {
//		// draw the circle center
//		circle( img2, cands[i].center*scaleby, 2, Scalar(0,255,0) , -1, 8, 0 );
//		// draw the circle outline
//		circle( img2, cands[i].center*scaleby, cands[i].radius*scaleby, Scalar(0,0,255), 1, 8, 0 );
//	}
//
//	namedWindow("circ", CV_GUI_EXPANDED);
//	imshow("circ", img2);
//	waitKey(0);
//	return Vec3f(0,0,0);
//}
//
//Vec3f findPetriDish3(Mat img) 
//{
//	static int cannyParam1 = 40;
//	static int maxSize = 1024;
//	static int minContourSize = 80;
//	static double minRadius = maxSize / 10;
//
//	// Convert to scaled grayscale image
//	Mat gray;
//	cvtColor(img, gray, CV_BGR2GRAY);
//
//	// Scale to 1024 size
//	double scaleby = max(gray.rows, gray.cols)*1.0/maxSize;
//	Mat resized;
//	resize(gray, resized, Size(), 1.0/scaleby, 1.0/scaleby, INTER_CUBIC); 
//	gray = resized;
//
//	timeit(NULL);
//
//	// Find edges 
//	Mat edges;
//	Canny(gray, edges, cannyParam1, cannyParam1/2);
//
//	// Find contours
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	findContours(edges, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//
//	// Remove small contours
//	vector<vector<Point> > contours2;
//	for (int c=0;c<contours.size();c++) {
//		Rect r = boundingRect(contours[c]);
//
//		if (r.width > minContourSize || r.height > minContourSize)
//			contours2.push_back(contours[c]);
//	}
//	contours=contours2;
//
//
//	// Gather all points
//	vector<Point> allpoints;
//	for (int c=0;c<contours.size();c++) {
//		for (int i=0;i<contours.size();i++) {
//			allpoints.push_back(contours[c][i]);
//		}
//	}
//
//	Point2f center;
//	float radius;
//
//	// Get initial estimate
//	minEnclosingCircle(allpoints, center, radius);
//	Vec3f circ = fitCircleToPoints(allpoints, Vec3f(center.x, center.y, radius));
//	CircCand cand;
//	cand.center = Point2f(circ[0], circ[1]);
//	cand.radius = circ[2];
//
//	timeit("GOT IT");
//
//	Mat img2 = img.clone();
//	// draw the circle center
//	circle( img2, cand.center*scaleby, 2, Scalar(0,255,0) , -1, 8, 0 );
//	// draw the circle outline
//	circle( img2, cand.center*scaleby, cand.radius*scaleby, Scalar(0,0,255), 1, 8, 0 );
//
//	namedWindow("circ", CV_GUI_EXPANDED);
//	imshow("circ", img2);
//	waitKey(0);
//	return Vec3f(0,0,0);
//}
