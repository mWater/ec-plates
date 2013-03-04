#include <opencv2/core/core.hpp>

#include "Circle.h"
#include "CircleFinder.h"
#include "ColonyCounter.h"
#include "OpenCVActivityContext.h"
#include "svm_params.h"

using namespace cv;
using namespace std;

void analyseECPlate(OpenCVActivityContext& context) {
	// Load image
	Mat img = imread(context.getParam(0));

	// Determine scale
	Mat screen = *context.getScreen();
	double scale = min(screen.rows*1.0 / img.rows, screen.cols*1.0 / img.cols);
	resize(img, screen, screen.size(), 0, 0, INTER_NEAREST);
	circle(screen, Point(100, 100), 20, Scalar(255, 255, 255, 255));
	context.updateScreen();

	// Find petri img
	Rect petriRect = findPetriRect(img);
	Mat petri = img(petriRect);

	ColonyCounter colonyCounter;
	// Read from string, not disk : 
	colonyCounter.loadTraining("svm_params.yml");
	//colonyCounter.loadTrainingString(svm_params);

	// Preprocess image
	petri = colonyCounter.preprocessImage(petri);

	if (context.getParamCount() >= 3) {
		imwrite(context.getParam(2), petri);
	}

	// Classify image
	Mat classified = colonyCounter.classifyImage(petri);

	// Count colonies
	int red, blue;
	Mat debugImage;
	colonyCounter.countColonies(classified, red, blue, true, &debugImage);

	if (context.getParamCount() >= 2) {
		imwrite(context.getParam(1), debugImage);
	}

	waitKey(0); //###
	context.setReturnValue(format("{\"red\": %d, \"blue\": %d, \"algorithm\": \"2013-02-14\"}", red, blue));
}
