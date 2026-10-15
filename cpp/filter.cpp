#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Point {
    std::array<double, 3> xyz;
    int label;
    double confidence;
};
std::vector<Point> static_points(const std::vector<Point>& points, double threshold=0.3) {
    if (threshold < 0 || threshold > 1) throw std::invalid_argument("confidence threshold");
    std::vector<Point> result;
    for (const auto& p : points)
        if (p.label != 10 && p.label != 11 && p.confidence >= threshold)
            result.push_back(p);
    return result;
}
int main() {
    const std::vector<Point> points={{{0,0,0},0,1},{{1,0,0},10,1}};
    const auto filtered=static_points(points);
    if (filtered.size()!=1) return 1;
    std::cout << filtered.size() << " static point retained\n";
}
