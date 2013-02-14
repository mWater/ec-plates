// Circle.cpp: implementation of the Circle class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Circle.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Circle::Circle()
{
	this->m_dRadius=-1;		// error checking 
}

Circle::~Circle()
{

}

Circle::Circle(Point pt1, Point pt2, Point pt3)
{
	this->m_dRadius=-1;		// error checking 

	if (!this->IsPerpendicular(pt1, pt2, pt3) )				this->CalcCircle(pt1, pt2, pt3);	
	else if (!this->IsPerpendicular(pt1, pt3, pt2) )		this->CalcCircle(pt1, pt3, pt2);	
	else if (!this->IsPerpendicular(pt2, pt1, pt3) )		this->CalcCircle(pt2, pt1, pt3);	
	else if (!this->IsPerpendicular(pt2, pt3, pt1) )		this->CalcCircle(pt2, pt3, pt1);	
	else if (!this->IsPerpendicular(pt3, pt2, pt1) )		this->CalcCircle(pt3, pt2, pt1);	
	else if (!this->IsPerpendicular(pt3, pt1, pt2) )		this->CalcCircle(pt3, pt1, pt2);	
	else { 
		this->m_dRadius=-1;
		return ;
	}
}

bool Circle::IsPerpendicular(Point pt1, Point pt2, Point pt3)
// Check the given point are perpendicular to x or y axis 
{
	double yDelta_a= pt2.y - pt1.y;
	double xDelta_a= pt2.x - pt1.x;
	double yDelta_b= pt3.y - pt2.y;
	double xDelta_b= pt3.x - pt2.x;
	

//	TRACE(" yDelta_a: %f xDelta_a: %f \n",yDelta_a,xDelta_a);
//	TRACE(" yDelta_b: %f xDelta_b: %f \n",yDelta_b,xDelta_b);

	// checking whether the line of the two pts are vertical
	if (fabs(xDelta_a) <= 0.000000001 && fabs(yDelta_b) <= 0.000000001){
		return false;
	}

	if (fabs(yDelta_a) <= 0.0000001){
		return true;
	}
	else if (fabs(yDelta_b) <= 0.0000001){
		return true;
	}
	else if (fabs(xDelta_a)<= 0.000000001){
		return true;
	}
	else if (fabs(xDelta_b)<= 0.000000001){
		return true;
	}
	else return false ;
}

double Circle::CalcCircle(Point2d pt1, Point2d pt2, Point2d pt3)
{
	double yDelta_a= pt2.y - pt1.y;
	double xDelta_a= pt2.x - pt1.x;
	double yDelta_b= pt3.y - pt2.y;
	double xDelta_b= pt3.x - pt2.x;
	
	if (fabs(xDelta_a) <= 0.000000001 && fabs(yDelta_b) <= 0.000000001){
		this->m_Center.x= 0.5*(pt2.x + pt3.x);
		this->m_Center.y= 0.5*(pt1.y + pt2.y);
		this->m_dRadius= norm(this->m_Center-pt1);		// calc. radius

		return this->m_dRadius;
	}
	
	// IsPerpendicular() assure that xDelta(s) are not zero
	double aSlope=yDelta_a/xDelta_a; // 
	double bSlope=yDelta_b/xDelta_b;
	if (fabs(aSlope-bSlope) <= 0.000000001){	// checking whether the given points are colinear. 	
		return -1;
	}

	// calc center
	this->m_Center.x= (aSlope*bSlope*(pt1.y - pt3.y) + bSlope*(pt1.x + pt2 .x)
		- aSlope*(pt2.x+pt3.x) )/(2* (bSlope-aSlope) );
	this->m_Center.y = -1*(this->m_Center.x - (pt1.x+pt2.x)/2)/aSlope +  (pt1.y+pt2.y)/2;

	this->m_dRadius= norm(this->m_Center - pt1);		// calc. radius
	return this->m_dRadius;
}

Point2d Circle::GetCenter()
{
	return this->m_Center;

}

double Circle::GetRadius()
{
	return this->m_dRadius;
}