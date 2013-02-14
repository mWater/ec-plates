#include "stdafx.h"
#include <opencv2/opencv.hpp>

#include "Circle.h"
#include "CircleFinder.h"
#include "ColonyCounter.h"

using namespace cv;

static int param1 = 160;
static int adapSize = 3;
static int adapC = 10;

static Mat image;

static double t = 0;

void timeit(const char *name) {
	if (name!=NULL) {
		printf("%s : %5.3f s\n", name, ((double)getTickCount() - t)/getTickFrequency());
	}
	t = (double)getTickCount();
}

void testCircles() 
{
	bool debug = false;
	for (int k=1;k<=4;k++) 
	{
		// Load image
		image = imread(format("images/%03d.jpg", k));
		Mat refImage = imread(format("images/%03d_circle.png", k));

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


static const int NUM_SAMPLES = 11;

//void batchExtractPetri(ColonyCounter& colonyCounter) 
//{
//	for (int k=1;k<=NUM_SAMPLES;k++)
//	{
//		// Extract petri
//		Mat img = imread(format("images/%03d.jpg", k));
//		Rect petriRect;
//		Mat petri = colonyCounter.preprocessImage(img, petriRect);
//		imwrite(format("images/output/%03d_petri.jpg", k), petri);
//
//		// Crop label if present
//		Mat labelImg = imread(format("images/train/%03d_label.png", k));
//		if (labelImg.rows == 0)
//			continue;
//		imwrite(format("images/output/%03d_petri_label.png", k), labelImg(petriRect));
//	}
//}

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

	printf("%s: Red=%3d vs %3d     Blue=%3d vs %3d    Error: %5.1f\n", path.c_str(), 
		red, redExpected, blue, blueExpected, error);


	bool notOk = false;

	// Check if within 20%
	if (redExpected >=0) 
	{
		if ((redExpected == 0 && red > 0)
			|| ((double)red)/redExpected > 1.2 || ((double)red)/redExpected < 0.8)
		{
			printf("########## RED NOT WITHIN TOLERANCES\n\n");
			//notOk = true;
		}
	}
	if ((blueExpected == 0 && blue > 0) 
		|| (blueExpected > 0) && (((double)blue)/blueExpected > 1.2 || ((double)blue)/blueExpected < 0.8))
	{
		printf("########## BLUE NOT WITHIN TOLERANCES\n\n");
		notOk = true;
	}
	if (notOk) {
		imshow(path, debugImage);
		imshow(path + " - petri", petri);
		waitKey(0);
	}
}

void runTests() 
{
	ColonyCounter colonyCounter;
	colonyCounter.loadTraining("svm_params.yml");

	FileStorage fs("tests.yml", FileStorage::READ);

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
		runTest(colonyCounter, path, red, blue, error);
		absErrorSum += fabs(error);

	}
	fs.release();

	printf("Error %f\n", absErrorSum);
}

int main(int argc, char* argv[])
{
	if (argc == 1) {
		char *appname = "ECPlates";
		printf("Usage:\n");
		printf(" %s count <image name> [<colony image file>] [<petri image file>]\nCounts colonies in an image, saving output to optional files\n\n", appname);
		printf(" %s train\nRun training (advanced)\n\n", appname);
		printf(" %s test\nRun tests (advanced)\n\n", appname);
		return 0;
	}

	if (strcmp(argv[1], "train") == 0) {
		vector<string> trainPaths;
		vector<string> labelPaths;
		for (int k=1;k<=NUM_SAMPLES;k++)
		{
			Mat label = imread(format("images/train/%03d_label.png", k));
			if (label.rows == 0)
				continue;

			trainPaths.push_back(format("images/%03d.jpg", k));
			labelPaths.push_back(format("images/train/%03d_label.png", k));
		}
		ColonyCounter colonyCounter;
		colonyCounter.trainClassifier(trainPaths, labelPaths);
		colonyCounter.saveTraining("svm_params.yml");
		return 0;
	}

	if (strcmp(argv[1], "test") == 0) {
		runTests();
	}

	if (strcmp(argv[1], "count") == 0) {
		// Load image
		Mat img = imread(argv[2]);

		// Find petri img
		Rect petriRect = findPetriRect(img);
		Mat petri = img(petriRect);

		ColonyCounter colonyCounter;
		colonyCounter.loadTraining("svm_params.yml");

		// Preprocess image
		petri = colonyCounter.preprocessImage(petri);

		if (argc >= 5) {
			imwrite(argv[4], petri);
		}

		// Classify image
		Mat classified = colonyCounter.classifyImage(petri);

		// Count colonies
		int red, blue;
		Mat debugImage;
		colonyCounter.countColonies(classified, red, blue, true, &debugImage);

		if (argc >= 4) {
			imwrite(argv[3], debugImage);
		}

		printf("{\"red\": %d, \"blue\": %d, \"algorithm\": \"2013-02-14\"}\n", red, blue);
	}

	return 0;
}
