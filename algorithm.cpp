#include <opencv2/core/core.hpp>

#include "Circle.h"
#include "CircleFinder.h"
#include "ColonyCounter.h"
#include "OpenCVActivityContext.h"
#include "svm_table.h"

#include <unistd.h>

using namespace cv;
using namespace std;

void analyseECPlate(OpenCVActivityContext& context) {
	context.log("Reading image");

	// Load image
	Mat img = imread(context.getParam(0));
	if (img.empty()) {
		context.setReturnValue("{\"error\":\"Image file not found\"}");
		return;
	}

	context.updateScreen(img);

	context.log("Finding petri image");

	// Find petri img
	Rect petriRect = findPetriRect(img);

	if (petriRect.height == 0) {
		context.log("Circle not found");
		context.setReturnValue("{\"error\":\"EC Plate not detected\"}");
		return;
	}

	Mat petri = img(petriRect);
	context.updateScreen(petri);

	context.log("Loading training");

	ColonyCounter colonyCounter;
	colonyCounter.loadTrainingQuantized(svmLookup, svmQuants);

	context.log("Preprocessing image");

	// Preprocess image
	petri = colonyCounter.preprocessImage(petri);
	context.updateScreen(petri);

	if (context.getParamCount() >= 3) {
		imwrite(context.getParam(2), petri);
	}

	context.log("Classifying image");

	// Classify image
	Mat debugImage;
	Mat classified = colonyCounter.classifyImage(petri, true, &debugImage);
	context.updateScreen(debugImage);

	context.log("Counting colonies");

	// Count colonies
	int red, blue;
	colonyCounter.countColonies(classified, red, blue, true, &debugImage);
	context.updateScreen(debugImage);

	if (context.getParamCount() >= 2) {
		imwrite(context.getParam(1), debugImage);
	}

	context.log("Showing results");

	// Pause
	sleep(2);

	context.log("Done");

	context.setReturnValue(format("{\"tc\": %d, \"ecoli\": %d, \"algorithm\": \"2013-03-15\"}", red, blue));
}
