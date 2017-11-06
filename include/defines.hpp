// defines.h
#ifndef _DEFINES_H
#define _DEFINES_H

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#define WARN(a)  std::cout << "WARNING[" << __FUNCTION__ <<" in line "<<__LINE__<<"]: "<< a << std::endl
#define ERR(a)  std::cout << "ERROR[" << __FUNCTION__ <<" in line "<<__LINE__<<"]: "<< a << std::endl
#define INFO(a)  std::cout << "INFO[" << __FUNCTION__ <<" in line "<<__LINE__<<"]: "<< a << std::endl
#define HEAD -1
#define INVALID -2
#define SINGLE "single"
#define STEREO "stereo"

#endif // _DEFINES_H