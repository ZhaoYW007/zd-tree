#pragma once

#include <algorithm>
#include <boost/random/linear_feedback_shift.hpp>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>

#include "common/geometryIO.h"
#include "common/parse_command_line.h"
#include "common/time_loop.h"
#include "parlay/internal/group_by.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"

#include "../benchmarks/nearestNeighbors/octTree/neighbors.h"
#include "common/IO.h"
#include "common/parse_command_line.h"
#include "parlay/random.h"

// using coord = long;
using coord = double;
using point2 = point2d<coord>;
using point3 = point3d<coord>;

std::pair<size_t, int> read_points_2d(const char* iFile, parlay::sequence<point2>& wp, int K) {
    parlay::sequence<char> S = readStringFromFile(iFile);
    parlay::sequence<char*> W = stringToWords(S);
    size_t N = std::stoul(W[0], nullptr, 10);
    int Dim = atoi(W[1]);
    assert(N >= 0 && Dim >= 1 && N >= K);

    auto pts = W.cut(2, W.size());
    assert(pts.size() % Dim == 0);
    size_t n = pts.size() / Dim;
    auto a = parlay::tabulate(Dim * n, [&](size_t i) -> coord {
        if constexpr (std::is_integral_v<coord>)
            return std::stol(pts[i]);
        else if (std::is_floating_point_v<coord>)
            return std::stod(pts[i]);
    });
    wp.resize(N);
    parlay::parallel_for(0, n, [&](size_t i) {
        wp[i].x = a[i * Dim];
        wp[i].x = a[i * Dim + 1];
    });
    return std::make_pair(N, Dim);
}

std::pair<size_t, int> read_points_3d(const char* iFile, parlay::sequence<point3>& wp, int K) {
    parlay::sequence<char> S = readStringFromFile(iFile);
    parlay::sequence<char*> W = stringToWords(S);
    size_t N = std::stoul(W[0], nullptr, 10);
    int Dim = atoi(W[1]);
    assert(N >= 0 && Dim >= 1 && N >= K);

    auto pts = W.cut(2, W.size());
    assert(pts.size() % Dim == 0);
    size_t n = pts.size() / Dim;
    auto a = parlay::tabulate(Dim * n, [&](size_t i) -> coord {
        if constexpr (std::is_integral_v<coord>)
            return std::stol(pts[i]);
        else if (std::is_floating_point_v<coord>)
            return std::stod(pts[i]);
    });
    wp.resize(N);
    parlay::parallel_for(0, n, [&](size_t i) {
        wp[i].x = a[i * Dim];
        wp[i].x = a[i * Dim + 1];
        wp[i].x = a[i * Dim + 2];
    });
    return std::make_pair(N, Dim);
}
