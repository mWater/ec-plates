#include "stdafx.h"
#include <opencv2/opencv.hpp>

#include "Circle.h"
#include "CircleFinder.h"
#include "ColonyCounter.h"
#include "OpenCVActivityContext.h"
#include "algorithm.h"
#include "svm_table.h"

using namespace cv;

/*
 * Main routine and associated support routines for running
 * commandline version of plate counter.
 */

static Mat image;

// Default quantizations to use
static int quants[] = { 256, 256 };

// Maximum number of sample images (numbered 001, 002, etc.)
// to use for training. Only those actually present will be used
static const int NUM_SAMPLES = 12;

static double t = 0;

void timeit(const char *name) {
	if (name!=NULL) {
		printf("%s : %5.3f s\n", name, ((double)getTickCount() - t)/getTickFrequency());
	}
	t = (double)getTickCount();
}

/*
 * Tests that circles are correctly found
 */
void runTestCircles()
{
	bool debug = false;
	for (int k=1;k<=4;k++) 
	{
		// Load image
		image = imread(format("samples/images/%03d.jpg", k));
		Mat refImage = imread(format("samples/train/%03d_circle.png", k));

		Vec3f circ = findPetriDish(image);
		testCirclePerformance(circ, refImage);

		if (debug) {
			circle(refImage, Point(circ[0], circ[1]), circ[2], Scalar(255,0,0), 2);
			namedWindow(format("%03d", k), CV_GUI_EXPANDED);
			imshow(format("%03d", k), refImage);
			waitKey(0);
		}
	}
}

/*
 * Runs a counting test on an image
 */
void runTest(ColonyCounter& colonyCounter, string path, int redExpected, int blueExpected, double &error) 
{
	// Load image
	Mat img = imread(path);

	// Find petri img
	Rect petriRect = findPetriRect(img);
	Mat petri = img(petriRect);

	// Preprocess image
	petri = colonyCounter.preprocessImage(petri);

	// Classify image
	Mat classified = colonyCounter.classifyImage(petri);

	// Count colonies
	int red, blue;
	Mat debugImage;
	colonyCounter.countColonies(classified, red, blue, true, &debugImage);

	// Set error as percentage +/-
	if (blueExpected == 0) {
		if (blue == 0)
			error = 0;
		else
			error = 100;
	}
	else {
		error = (((double)blue)/blueExpected - 1) * 100;
	}

	bool redNotOk = false, blueNotOk = false;

	// Check if within 20%
	if (redExpected >=0) 
	{
		if ((redExpected == 0 && red > 0)
			|| ((double)red)/redExpected > 1.2 || ((double)red)/redExpected < 0.8)
		{
			redNotOk = true;
		}
	}
	if ((blueExpected == 0 && blue > 0) 
		|| ((blueExpected > 0) && (((double)blue)/blueExpected > 1.2 || ((double)blue)/blueExpected < 0.8)))
	{
		blueNotOk = true;
	}

	printf("[%6s] %s:   Blue=%3d (%3d)%s    Red=%3d (%3d)%s\n",
		blueNotOk || redNotOk ? "wrong" : "ok",
		path.c_str(),
		blue, blueExpected, blueNotOk ? "*" : " ",
		red, redExpected, redNotOk ? "*" : " ");

//	if (blueNotOk) {
//		imshow(path, debugImage);
//		imshow(path + " - petri", petri);
//		waitKey(0);
//	}
}

/*
 * Runs all counting tests, reading tests.yml from the samples folder
 * to determine which images to count and what the expected values are
 */
void runTests() 
{
	ColonyCounter colonyCounter;
	colonyCounter.loadTraining("svm_params.yml");

	FileStorage fs("samples/tests.yml", FileStorage::READ);

	double absErrorSum = 0;

	FileNode features = fs["tests"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;
	for( ; it != it_end; ++it, idx++ )
	{
		string path;
		(*it)["path"] >> path;
		int red = (int)(*it)["red"];
		int blue = (int)(*it)["blue"];

		double error;
		runTest(colonyCounter, "samples/" + path, red, blue, error);
		absErrorSum += fabs(error);

	}
	fs.release();

	printf("Error %f\n", absErrorSum);
}

/*
 * Like runTests, but using the quantized lookup table
 * instead of the support vector machine directly.
 */
void runTestsSVMTable()
{
	ColonyCounter colonyCounter;
	colonyCounter.loadTrainingQuantized(svmLookup, svmQuants);

	FileStorage fs("samples/tests.yml", FileStorage::READ);

	double absErrorSum = 0;

	FileNode features = fs["tests"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;
	for( ; it != it_end; ++it, idx++ )
	{
		string path;
		(*it)["path"] >> path;
		int red = (int)(*it)["red"];
		int blue = (int)(*it)["blue"];

		double error;
		runTest(colonyCounter, "samples/" + path, red, blue, error);
		absErrorSum += fabs(error);

	}
	fs.release();

	printf("Error %f\n", absErrorSum);
}

/*
 * Run tests to make sure that quantization is working.
 */
void runQuantTests()
{
	ColonyCounter colonyCounter;
	colonyCounter.loadTraining("svm_params.yml");

	FileStorage fs("samples/tests.yml", FileStorage::READ);

	FileNode features = fs["tests"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;
	for( ; it != it_end; ++it, idx++ )
	{
		string path;
		(*it)["path"] >> path;

		Mat img = imread("samples/" + path);

		// Find petri img
		Rect petriRect = findPetriRect(img);
		Mat petri = img(petriRect);

		// Preprocess image
		petri = colonyCounter.preprocessImage(petri);

		colonyCounter.testQuantization(petri, quants);

		// Classify image
		Mat debugImg, debugImgq;
		colonyCounter.classifyImage(petri, true, &debugImg);
		colonyCounter.classifyImageQuant(petri, true, &debugImgq, quants);
		imshow("normal", debugImg);
		imshow("quant", debugImgq);
		waitKey(0);
	}
	fs.release();
}

int main(int argc, char* argv[])
{
	if (argc == 1) {
		char *appname = "ECPlates";
		printf("Usage:\n");
		printf(" %s count <image name> [<colony image file>] [<petri image file>]\nCounts colonies in an image, saving output to optional files\n\n", appname);
		printf(" %s count-gui <image name> [<colony image file>] [<petri image file>]\nCounts colonies in an image with a gui, saving output to optional files\n\n", appname);
		printf(" %s train\nRun training (advanced)\n\n", appname);
		printf(" %s test\nRun tests (advanced)\n\n", appname);
		printf(" %s testq\nRun tests using quantized lookup table (advanced)\n\n", appname);
		printf(" %s quant\nRun quantization tests (advanced)\n\n", appname);
		printf(" %s test-circles\nRun circle tests (advanced)\n\n", appname);
		return 0;
	}

	if (strcmp(argv[1], "quant") == 0) {
		runQuantTests();
	}

	if (strcmp(argv[1], "train") == 0) {
		vector<string> trainPaths;
		vector<string> labelPaths;
		for (int k=1;k<=NUM_SAMPLES;k++)
		{
			Mat label = imread(format("samples/train/%03d_label.png", k));
			if (label.rows == 0)
				continue;

			trainPaths.push_back(format("samples/images/%03d.jpg", k));
			labelPaths.push_back(format("samples/train/%03d_label.png", k));
		}
		ColonyCounter colonyCounter;
		colonyCounter.trainClassifier(trainPaths, labelPaths, NULL);
		colonyCounter.saveTraining("svm_params.yml");
		colonyCounter.saveTrainingQuantized("svm_table.h", quants);
		return 0;
	}

	if (strcmp(argv[1], "test") == 0) {
		runTests();
	}

	if (strcmp(argv[1], "testq") == 0) {
		runTestsSVMTable();
	}

	if (strcmp(argv[1], "test-circles") == 0) {
		runTestCircles();
	}

	if (strcmp(argv[1], "count") == 0) {
		ConsoleOpenCVActivityContext context(argc-2, argv+2, false);
		analyseECPlate(context);
		printf("%s\n", context.returnValue.c_str());
	}

	if (strcmp(argv[1], "count-gui") == 0) {
		DesktopOpenCVActivityContext context(argc-2, argv+2);
		analyseECPlate(context);
		printf("%s\n", context.returnValue.c_str());
		waitKey(0);
	}

	return 0;
}
