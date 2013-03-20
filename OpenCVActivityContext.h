#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
#include <stdio.h>

#pragma once

/*
 * Interface through which the algorithm interacts with the outside world,
 * allowing it to be used from the command line, within a GUI or
 * embedded in an Android application.
 */
class OpenCVActivityContext {
public:
	virtual ~OpenCVActivityContext() {}

	// Get parameters (all of which are strings) that may optionally be passed to the algorithm
	virtual string getParam(int n) = 0;
	virtual int getParamCount() = 0;

	// Set the return value of the algorithm
	virtual void setReturnValue(string val) = 0;

	// If supported, display the image specified to the screen
	virtual void updateScreen(Mat& screen) = 0;

	// If supported, log the specified message
	virtual void log(string msg) = 0;

	// Check if the user has aborted the operation
	virtual bool isAborted() = 0;
};

/*
 * Context designed for commandline use only with no GUI
 */
class ConsoleOpenCVActivityContext : public OpenCVActivityContext {
public:
	ConsoleOpenCVActivityContext(int argc, char* argv[], bool logging) :
		argc(argc), argv(argv), logging(logging) {
	}

	~ConsoleOpenCVActivityContext() {
	}

	string getParam(int n) {
		return argv[n];
	}

	int getParamCount() {
		return argc;
	}

	void setReturnValue(string val) {
		returnValue = val;
	}

	void updateScreen(Mat& screen) {
	}

	void log(string msg) {
		if (logging)
			printf("%s\n", msg.c_str());
	}

	bool isAborted() {
		return false;
	}

	string returnValue;

private:
	int argc;
	char** argv;
	bool logging;
};

/*
 * Context designed for a program run on the desktop with a GUI
 */
class DesktopOpenCVActivityContext : public OpenCVActivityContext {
public:
	DesktopOpenCVActivityContext(int argc, char* argv[]) :
		argc(argc), argv(argv) {
	}

	~DesktopOpenCVActivityContext() {
	}

	string getParam(int n) {
		return argv[n];
	}

	int getParamCount() {
		return argc;
	}

	void setReturnValue(string val) {
		returnValue = val;
	}

	void updateScreen(Mat& screen) {
		imshow("screen", screen);
		waitKey(0);
	}

	void log(string msg) {
		printf("%s\n", msg.c_str());
	}

	bool isAborted() {
		return false;
	}

	string returnValue;

private:
	int argc;
	char** argv;
};


