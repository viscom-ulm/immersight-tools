#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace viscom {

	using Real = float;
	using Point = cv::Point2f;
	using Point3 = cv::Point3f;
	using Edge = std::pair<Point, Point>;
	template <typename T>
	using List = std::vector<T>;

	bool inside(const Edge &edge, const Point &p)
	{
		/* vertices of the clip polygon are consistently listed in a counter-clockwise direction */
		/* http://blackpawn.com/texts/pointinpoly/ */
		Point3 AB = Point3(edge.second - edge.first);
		Point3 Ap = Point3(p - edge.first);
		Point3 crossproduct = AB.cross(Ap);
		return crossproduct.z > 0;
	}

	bool rayIntersection(const Point &a1, const Point &a2, const Point &b1, const Point &b2, Point &intersecPoint) {
		Point p = a1;
		Point q = b1;
		Point r(a2 - a1);
		Point s(b2 - b1);

		if (r.cross(s) < /*EPS*/1e-8) { return false; }
		float t = (q - p).cross(s) / r.cross(s);
		intersecPoint = p + t * r;
		return true;
	}

	bool lineIntersection(const Point &a1, const Point &a2, const Point &b1, const Point &b2, Real *t0, Real *t1, bool *intoA, Point & point) {
		Point p = a1;
		Point q = b1;
		Point r(a2 - a1);
		Point s(b2 - b1);

		if (abs(r.cross(s)) < /*EPS*/1e-8) { return false; }
		*t0 = (q - p).cross(s) / r.cross(s);
		*t1 = (p - q).cross(r) / s.cross(r);
		// check travel direction of ray b, relative to edge a
		*intoA = inside(std::make_pair(a1, a2), b2);

		if (*t0 >= 0.0f && *t0 <= 1.0f) {
			if (*t1 >= 0.0f && *t1 <= 1.0f) {
				point = p + *t0 * r;
				return true;
			}
		}
		return false;
	}

	bool ComputeIntersection(const Point &S, const Point &E, const Edge &clipEdge, Point &intersecPoint)
	{
		return rayIntersection(S, S - E, clipEdge.first, clipEdge.first - clipEdge.second, intersecPoint);
	}

	void SutherlandHodgmanAlgorithmn(const List<Point> &subjectPolygon, const List<Edge> & clipPolygon, List<Point> &outputList) {
		/* https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm */
		outputList = subjectPolygon;
		Point S, E, intersecPoint;
		for (size_t i = 0; i < clipPolygon.size(); i++) {
			Edge clipEdge = clipPolygon[i];
			List<Point> inputList = outputList;
			outputList.clear();
			S = inputList.back();
			for (size_t j = 0; j < inputList.size(); j++) {
				E = inputList[j];
				if (inside(clipEdge, E)) {
					if (!inside(clipEdge, S)) {
						if (ComputeIntersection(S, E, clipEdge, intersecPoint)) {
							outputList.push_back(intersecPoint);
						}
					}
					outputList.push_back(E);
				}
				else if (inside(clipEdge, S)) {
					if (ComputeIntersection(S, E, clipEdge, intersecPoint)) {
						outputList.push_back(intersecPoint);
					}
				}
				S = E;
			}
		}
	}
}