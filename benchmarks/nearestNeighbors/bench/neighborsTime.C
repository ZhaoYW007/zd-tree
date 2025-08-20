// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <iostream>
#include <cstdint>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstddef>

#include "../benchmarks/nearestNeighbors/octTree/neighbors.h"
#include "common/IO.h"
#include "common/parse_command_line.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include "utils/random_generator.h"
#include "testFramework.h"

#ifdef USE_PAPI
#include "utils/papi.h"
#endif

using namespace benchIO;
using namespace std;

// *************************************************************
//  SOME DEFINITIONS
// *************************************************************

using coord = double;
using point2 = point2d<coord>;
using point3 = point3d<coord>;

template<class PT, int KK>
struct vertex {
    using pointT = PT;
    int identifier;
    pointT pt;        // the point itself
    vertex *ngh[KK];  // the list of neighbors
    vertex(pointT p, int id = 0) : pt(p), identifier(id) {}
    vertex(int id = 0) : identifier(id) {}
    size_t counter;
    size_t counter2;
};

/* Argv for main function */
int64_t total_insert_size;
int NR_DIMENSION;
int test_batch_size;
int test_round;
int test_type; /* 1: Point search; 2: Box range count; 3: Box fetch; 4: kNN */
int expected_box_size;
std::string file_name;

void host_parse_arguments(int argc, char *argv[]) {
    file_name          = (argc >= 2  ?      argv[1]    : "uniform");
    NR_DIMENSION       = (argc >= 3  ? stoi(argv[2] )  : 3       );
    total_insert_size  = (argc >= 4  ? stoi(argv[3] )  : 500000  );
    test_type          = (argc >= 5  ? stoi(argv[4] )  : 0       );
    test_batch_size    = (argc >= 6  ? stoi(argv[5] )  : 10000   );
    test_round         = (argc >= 7  ? stoi(argv[6] )  : 2       );
    expected_box_size  = (argc >= 8  ? stoi(argv[7] )  : 100     );
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char *argv[]) {
    printf("------------------- Start ---------------------\n");
    host_parse_arguments(argc, argv);
    srand(0);
    rn_gen::init();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    double avg_time = 0.0;

    printf("------------- Data Structure Init ------------\n");

    using vtx2 = vertex<point2, 1>;
    using vtx3 = vertex<point3, 1>;

    parlay::sequence<point2> vectors_from_file_2d(1);
    parlay::sequence<vtx2> vectors_to_insert_2d(1);
    parlay::sequence<point3> vectors_from_file_3d(1);
    parlay::sequence<vtx3> vectors_to_insert_3d(1);
    size_t varden_counter = 0;
    coord COORD_MAX = INT64_MAX;

    if(file_name == "uniform") {
        printf("Uniform\n");
        if(NR_DIMENSION == 2) {
            vectors_to_insert_2d.resize(total_insert_size);
            parlay::parallel_for(0, total_insert_size, [&](size_t i) {
                vectors_to_insert_2d[i].pt.x = abs(rn_gen::parallel_rand());
                vectors_to_insert_2d[i].pt.y = abs(rn_gen::parallel_rand());
                vectors_to_insert_2d[i].identifier = i;
            });
        }
        else if(NR_DIMENSION == 3) {
            vectors_to_insert_3d.resize(total_insert_size);
            parlay::parallel_for(0, total_insert_size, [&](size_t i) {
                vectors_to_insert_3d[i].pt.x = abs(rn_gen::parallel_rand());
                vectors_to_insert_3d[i].pt.y = abs(rn_gen::parallel_rand());
                vectors_to_insert_3d[i].pt.z = abs(rn_gen::parallel_rand());
                vectors_to_insert_3d[i].identifier = i;
            });
        }
    }
    else {
        printf("File: %s\n", file_name.c_str());
        if(NR_DIMENSION == 2) {
            read_points_2d(file_name.c_str(), vectors_from_file_2d, 100);
            vectors_to_insert_2d = parlay::tabulate(total_insert_size, [&](size_t i) {
                return vtx2(vectors_from_file_2d[i], i);
            });
            if(test_type == 2 || test_type == 3) {
                COORD_MAX = parlay::reduce(parlay::delayed_tabulate(vectors_from_file_2d.size(), [&](size_t j) {
                    return std::max(vectors_from_file_2d[j].x, vectors_from_file_2d[j].y);
                }), parlay::maximum<coord>()) - parlay::reduce(parlay::delayed_tabulate(vectors_from_file_2d.size(), [&](size_t j) {
                    return std::min(vectors_from_file_2d[j].x, vectors_from_file_2d[j].y);
                }), parlay::minimum<coord>());
            }
        }
        else if(NR_DIMENSION == 3) {
            read_points_3d(file_name.c_str(), vectors_from_file_3d, 100);
            vectors_to_insert_3d = parlay::tabulate(total_insert_size, [&](size_t i) {
                return vtx3(vectors_from_file_3d[i], i);
            });
            if(test_type == 2 || test_type == 3) {
                COORD_MAX = parlay::reduce(parlay::delayed_tabulate(vectors_from_file_3d.size(), [&](size_t j) {
                    return std::max(vectors_from_file_3d[j].x, std::max(vectors_from_file_3d[j].y, vectors_from_file_3d[j].z));
                }), parlay::maximum<coord>()) - parlay::reduce(parlay::delayed_tabulate(vectors_from_file_3d.size(), [&](size_t j) {
                    return std::min(vectors_from_file_3d[j].x, std::min(vectors_from_file_3d[j].y, vectors_from_file_3d[j].z));
                }), parlay::minimum<coord>());
            }
        }
    }
    printf("------------- Finish Data Init ------------\n");

    using knn_tree_2d = k_nearest_neighbors<vtx2, 100>;
    using knn_tree_3d = k_nearest_neighbors<vtx3, 100>;
    using box_2d = knn_tree_2d::box;
    using box_3d = knn_tree_3d::box;
    using node_2d = knn_tree_2d::node;
    using node_3d = knn_tree_3d::node;
    using box_delta_2d = std::pair<box_2d, double>;
    using box_delta_3d = std::pair<box_3d, double>;

    parlay::sequence<vtx2*> v_input_2d(1, NULL);
    parlay::sequence<vtx3*> v_input_3d(1, NULL);
    knn_tree_2d T_2d(v_input_2d);
    knn_tree_3d T_3d(v_input_3d);
    if(NR_DIMENSION == 2) {
        v_input_2d = parlay::tabulate(vectors_to_insert_2d.size(), [&](size_t i) { return &vectors_to_insert_2d[i]; });
        box_2d whole_box = knn_tree_2d::o_tree::get_box(v_input_2d);
        knn_tree_2d T_2d = knn_tree_2d(v_input_2d, whole_box);
    }
    else if(NR_DIMENSION == 3) {
        v_input_3d = parlay::tabulate(vectors_to_insert_3d.size(), [&](size_t i) { return &vectors_to_insert_3d[i]; });
        box_3d whole_box = knn_tree_3d::o_tree::get_box(v_input_3d);
        knn_tree_3d T_3d = knn_tree_3d(v_input_3d, whole_box);
    }
    printf("------------- Finish Tree Build ------------\n");

#ifdef USE_PAPI
    papi_init_program(parlay::num_workers());
#endif

    if(test_type == 1) {
        printf("------------- Insert ------------\n");
        parlay::sequence<vtx2> vec_to_search_2d;
        parlay::sequence<vtx3> vec_to_search_3d;
        for(int i = 0, offset = total_insert_size; i < test_round; i++, offset += test_batch_size) {
            printf("Round: %d; Time: ", i);
            if(file_name == "uniform") {
                if(NR_DIMENSION == 2) vec_to_search_2d.resize(test_batch_size);
                else if(NR_DIMENSION == 3) vec_to_search_3d.resize(test_batch_size);
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    if(NR_DIMENSION == 2) {
                        vec_to_search_2d[j].pt.x = abs(rn_gen::parallel_rand());
                        vec_to_search_2d[j].pt.y = abs(rn_gen::parallel_rand());
                        vec_to_search_2d[j].identifier = j;
                    }
                    else if(NR_DIMENSION == 3) {
                        vec_to_search_3d[j].pt.x = abs(rn_gen::parallel_rand());
                        vec_to_search_3d[j].pt.y = abs(rn_gen::parallel_rand());
                        vec_to_search_3d[j].pt.z = abs(rn_gen::parallel_rand());
                        vec_to_search_3d[j].identifier = j;
                    }
                });
            }
            else {
                if(NR_DIMENSION == 2) {
                    vec_to_search_2d = parlay::tabulate(test_batch_size, [&](size_t j) {
                        return vtx2(vectors_from_file_2d[offset + j], j);
                    });
                }
                else if(NR_DIMENSION == 3) {
                    vec_to_search_3d = parlay::tabulate(test_batch_size, [&](size_t j) {
                        return vtx3(vectors_from_file_3d[offset + j], j);
                    });
                }
            }
#ifdef USE_PAPI
            papi_reset_counters();
            papi_turn_counters(true);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(true, parlay::num_workers());
#endif
            start_time = std::chrono::high_resolution_clock::now();
            if(NR_DIMENSION == 2) {
                node_2d *root = T_2d.get_root();
                box_delta_2d bd = T_2d.get_box_delta(2);
                auto v_insert_2d = parlay::tabulate(test_batch_size, [&](size_t j) { return &vec_to_search_2d[j]; });
                T_2d.batch_insert(vec_to_search_2d, root, bd.first, bd.second);
            }
            else if(NR_DIMENSION == 3) {
                node_3d *root = T_3d.get_root();
                box_delta_3d bd = T_3d.get_box_delta(3);
                auto v_insert_3d = parlay::tabulate(test_batch_size, [&](size_t j) { return &vec_to_search_3d[j]; });
                T_3d.batch_insert(vec_to_search_3d, root, bd.first, bd.second);
            }
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            printf("%f\n", d.count());
            avg_time += d.count();
        }
    }
    else if(test_type == 2 || test_type == 3) {
        printf("------------- Box ------------\n");
        double box_edge_size = COORD_MAX / pow(total_insert_size / expected_box_size, 1.0 / NR_DIMENSION) / 2.0;
        parlay::sequence<box_2d> boxes_2d;
        parlay::sequence<box_3d> boxes_3d;
        for(int i = 0, offset = total_insert_size; i < test_round; i++, offset += test_batch_size) {
            printf("Round: %d; Time: ", i);
            if(file_name == "uniform") {
                if(NR_DIMENSION == 2) boxes_2d.resize(test_batch_size);
                else if(NR_DIMENSION == 3) boxes_3d.resize(test_batch_size);
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    if(NR_DIMENSION == 2) {
                        boxes_2d[j].first.x = abs(rn_gen::parallel_rand());
                        boxes_2d[j].first.y = abs(rn_gen::parallel_rand());
                        boxes_2d[j].second.x = boxes_2d[j].first.x + box_edge_size * 2;
                        boxes_2d[j].second.y = boxes_2d[j].first.y + box_edge_size * 2;
                    }
                    else if(NR_DIMENSION == 3) {
                        boxes_3d[j].first.x = abs(rn_gen::parallel_rand());
                        boxes_3d[j].first.y = abs(rn_gen::parallel_rand());
                        boxes_3d[j].first.z = abs(rn_gen::parallel_rand());
                        boxes_3d[j].second.x = boxes_3d[j].first.x + box_edge_size * 2;
                        boxes_3d[j].second.y = boxes_3d[j].first.y + box_edge_size * 2;
                        boxes_3d[j].second.z = boxes_3d[j].first.z + box_edge_size * 2;
                    }
                });
            }
            else {
                if(NR_DIMENSION == 2) boxes_2d = parlay::tabulate(test_batch_size, [&](size_t j) {
                    point2d pnt = vectors_from_file_2d[offset + j];
                    pnt.x += box_edge_size * 2;
                    pnt.y += box_edge_size * 2;
                    return std::make_pair(vectors_from_file_2d[offset + j], pnt);
                });
                else if(NR_DIMENSION == 3) boxes_3d = parlay::tabulate(test_batch_size, [&](size_t j) {
                    point3d pnt = vectors_from_file_3d[offset + j];
                    pnt.x += box_edge_size * 2;
                    pnt.y += box_edge_size * 2;
                    pnt.z += box_edge_size * 2;
                    return std::make_pair(vectors_from_file_3d[offset + j], pnt);
                });
            }
#ifdef USE_PAPI
            papi_reset_counters();
            papi_turn_counters(true);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(true, parlay::num_workers());
#endif
            start_time = std::chrono::high_resolution_clock::now();
            if(NR_DIMENSION == 2) {
                node_2d *root = T_2d.get_root();
                box_delta_2d bd = T_2d.get_box_delta(2);
                for(size_t j = 0; j < test_batch_size; j++) {
                    T.range_count(root, boxes_2d[j], bd.second);
                    if(test_type == 3) {
                        parlay::sequence<vtx2*> out(root->get_aug(), nullptr);
                        T.range_query(root, parlay::make_slice(out), boxes_2d[j], bd.second);
                    }
                }
            }
            else if(NR_DIMENSION == 3) {
                node_3d *root = T_3d.get_root();
                box_delta_3d bd = T_3d.get_box_delta(3);
                for(size_t j = 0; j < test_batch_size; j++) {
                    T.range_count(root, boxes_3d[j], bd.second);
                    if(test_type == 3) {
                        parlay::sequence<vtx3*> out(root->get_aug(), nullptr);
                        T.range_query(root, parlay::make_slice(out), boxes_3d[j], bd.second);
                    }
                }
            }
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            printf("%f\n", d.count());
            avg_time += d.count();
        }
    }
    else if(test_type == 4) {
        printf("------------- kNN ------------\n");
        parlay::sequence<vtx2> vec_to_search_2d;
        parlay::sequence<vtx3> vec_to_search_3d;
        for(int i = 0, offset = total_insert_size; i < test_round; i++, offset += test_batch_size) {
            printf("Round: %d; Time: ", i);
            if(file_name == "uniform") {
                if(NR_DIMENSION == 2) vec_to_search_2d.resize(test_batch_size);
                else if(NR_DIMENSION == 3) vec_to_search_3d.resize(test_batch_size);
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    if(NR_DIMENSION == 2) {
                        vec_to_search_2d[j].pt.x = abs(rn_gen::parallel_rand());
                        vec_to_search_2d[j].pt.y = abs(rn_gen::parallel_rand());
                        vec_to_search_2d[j].identifier = j;
                    }
                    else if(NR_DIMENSION == 3) {
                        vec_to_search_3d[j].pt.x = abs(rn_gen::parallel_rand());
                        vec_to_search_3d[j].pt.y = abs(rn_gen::parallel_rand());
                        vec_to_search_3d[j].pt.z = abs(rn_gen::parallel_rand());
                        vec_to_search_3d[j].identifier = j;
                    }
                });
            }
            else {
                if(NR_DIMENSION == 2) {
                    vec_to_search_2d = parlay::tabulate(test_batch_size, [&](size_t j) {
                        return vtx2(vectors_from_file_2d[offset + j], j);
                    });
                }
                else if(NR_DIMENSION == 3) {
                    vec_to_search_3d = parlay::tabulate(test_batch_size, [&](size_t j) {
                        return vtx3(vectors_from_file_3d[offset + j], j);
                    });
                }
            }
#ifdef USE_PAPI
            papi_reset_counters();
            papi_turn_counters(true);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(true, parlay::num_workers());
#endif
            start_time = std::chrono::high_resolution_clock::now();
            if(NR_DIMENSION == 2) {
                node_2d *root = T_2d.get_root();
                box_delta_2d bd = T_2d.get_box_delta(2);
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    T_2d.k_nearest(&vec_to_search_2d[j], expected_box_size);
                });
            }
            else if(NR_DIMENSION == 3) {
                node_3d *root = T_3d.get_root();
                box_delta_3d bd = T_3d.get_box_delta(3);
                parlay::parallel_for(0, test_batch_size, [&](size_t j) {
                    T_3d.k_nearest(&vec_to_search_3d[j], expected_box_size);
                });
            }
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::
            end_time = std::chrono::high_resolution_clock::now();
#ifdef USE_PAPI
            papi_turn_counters(false);
            papi_check_counters(parlay::worker_id());
            papi_wait_counters(false, parlay::num_workers());
#endif
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            printf("%f\n", d.count());
            avg_time += d.count();
        }
    }
#ifdef USE_PAPI
    papi_print_counters(1);
#endif
    avg_time /= test_round;
    printf("Average Time: %f\n", avg_time);
    return 0;
}
