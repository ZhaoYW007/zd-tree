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

bool report_stats = false;
int algorithm_version = 0;
// 0=root based, 1=bit based, >2=map based

#include <math.h>

#include <algorithm>
#include <queue>

#include "common/geometry.h"
#include "common/geometryIO.h"
#include "common/time_loop.h"
#include "k_nearest_neighbors.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

//* export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 *//

// find the k nearest neighbors for all points in tree
// places pointers to them in the .ngh field of each vertex
template <int max_k, class vtx>
void ANN( parlay::sequence<vtx>& v, int k, int rounds,
          parlay::sequence<vtx>& vin, int tag, int queryType ) {
  //  timer t( "ANN", report_stats );

  {
    timer _t;
    _t.start();
    using knn_tree = k_nearest_neighbors<vtx, max_k>;
    using node = typename knn_tree::node;
    using box = typename knn_tree::box;
    using box_delta = std::pair<box, double>;

    // create sequences for insertion and deletion
    size_t n = v.size();
    parlay::sequence<vtx*> v2, vin2, allv;

    v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
    box whole_box = knn_tree::o_tree::get_box( v2 );
    knn_tree T = knn_tree( v2, whole_box );

    //* build
    double aveBuild = time_loop(
        rounds, 1.0, [&]() {},
        [&]() {
          v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
          whole_box = knn_tree::o_tree::get_box( v2 );
          T = knn_tree( v2, whole_box );
        },
        [&]() { T.tree.reset(); } );
    std::cout << aveBuild << " " << std::flush;

    //* restore
    v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
    whole_box = knn_tree::o_tree::get_box( v2 );
    T = knn_tree( v2, whole_box );

    // prelims for insert/delete
    int dims;
    node* root;
    box_delta bd;
    // parlay::sequence<vtx*> vin2 = parlay::sequence<vtx*>( vin.size() );

    //* batch-dynamic insertion
    if( tag >= 1 ) {
      double aveInsert = time_loop(
          rounds, 1.0,
          [&]() {
            T.tree.reset();
            v2 = parlay::tabulate( n,
                                   [&]( size_t i ) -> vtx* { return &v[i]; } );
            vin2 = parlay::tabulate(
                n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
            allv = parlay::append( v2, vin2 );
            whole_box = knn_tree::o_tree::get_box( allv );
            T = knn_tree( v2, whole_box );
          },
          [&]() {
            vin2 = parlay::tabulate(
                n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
            dims = vin2[0]->pt.dimension();
            root = T.tree.get();
            bd = T.get_box_delta( dims );

            T.batch_insert( vin2, root, bd.first, bd.second );
          },
          [&]() { T.tree.reset(); } );
      std::cout << aveInsert << " " << std::flush;

      //* restore
      T.tree.reset();
      v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
      vin2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
      allv = parlay::append( v2, vin2 );
      whole_box = knn_tree::o_tree::get_box( allv );
      T = knn_tree( v2, whole_box );
      dims = vin2[0]->pt.dimension();
      root = T.tree.get();
      bd = T.get_box_delta( dims );
      T.batch_insert( vin2, root, bd.first, bd.second );
      //! no need to append vin since KNN graph always get points from the
      //! tree
    }

    if( tag >= 2 ) {
      double aveDelete = time_loop(
          rounds, 1.0,
          [&]() {
            T.tree.reset();
            v2 = parlay::tabulate( n,
                                   [&]( size_t i ) -> vtx* { return &v[i]; } );
            vin2 = parlay::tabulate(
                n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
            allv = parlay::append( v2, vin2 );

            whole_box = knn_tree::o_tree::get_box( allv );
            //* warning not the same as others
            T = knn_tree( allv, whole_box );
            //  T = knn_tree( v2, whole_box );

            dims = vin2[0]->pt.dimension();
            root = T.tree.get();
            bd = T.get_box_delta( dims );
            //  T.batch_insert( vin2, root, bd.first, bd.second );
          },
          [&]() {
            vin2 = parlay::tabulate(
                n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
            T.batch_delete( vin2, root, bd.first, bd.second );
          },
          [&]() { T.tree.reset(); } );
      std::cout << aveDelete << " " << std::flush;

      T.tree.reset();
      v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
      vin2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
      allv = parlay::append( v2, vin2 );
      whole_box = knn_tree::o_tree::get_box( allv );
      //* warning not the same as others
      T = knn_tree( allv, whole_box );
      dims = vin2[0]->pt.dimension();
      root = T.tree.get();
      bd = T.get_box_delta( dims );
      T.batch_delete( vin2, root, bd.first, bd.second );
    }

    if( queryType & ( 1 << 0 ) ) {  //* KNN
      int K[3] = { 1, 10, 100 };
      for( int i = 0; i < 3; i++ ) {
        auto aveQuery = time_loop(
            rounds, 1.0,
            [&]() {

            },
            [&]() {
              k = K[i];
              if( algorithm_version == 0 ) {  // this is for starting from
                parlay::sequence<vtx*> vr = T.vertices();
                // find nearest k neighbors for each point
                //  vr = T.vertices();
                //  vr = parlay::random_shuffle( vr.cut( 0, vr.size() ) );
                size_t n = vr.size();
                parlay::parallel_for(
                    0, n, [&]( size_t i ) { T.k_nearest( vr[i], k ); } );
              } else if( algorithm_version == 1 ) {
              } else {
              }
            },
            [&]() {} );
        std::cout << aveQuery << " -1 -1 " << std::flush;
      }
    }

    // TODO rewrite range count
    int queryNum = 100;
    if( queryType & ( 1 << 1 ) ) {  //* range Count
      double aveQuery = time_loop(
          rounds, 1.0, [&]() {},
          [&]() {
            // for( int i = 0; i < 10; i++ ) {
            //   parlay::sequence<vtx*> pts{ v[i], v[( i + n / 2 ) % size] };
            //   box queryBox = knn_tree::o_tree::get_box( pts );
            //   int dims = ( v[0]->pt ).dimension();
            //   box_delta bd = T.get_box_delta( dims );
            //   T.range_count( T.tree.get(), queryBox, 1e-7 );
            //   //  std::cout << T.tree.get()->get_aug() << std::endl;
            // }
          },
          [&]() {} );
      std::cout << "-1 -1 -1 " << std::flush;
    }

    // if( queryType & ( 1 << 2 ) ) {  //* range query
    //   // parlay::sequence<vtx*> Out( size );
    //   // double aveQuery = time_loop(
    //   //     rounds, 1.0, [&]() {},
    //   //     [&]() {
    //   //       for( int i = 0; i < 10; i++ ) {
    //   //         parlay::sequence<vtx*> pts{ v[i], v[( i + size / 2 ) %
    //   size]
    //   };
    //   //         box queryBox = knn_tree::o_tree::get_box( pts );
    //   //         int dims = ( v[0]->pt ).dimension();
    //   //         box_delta bd = T.get_box_delta( dims );
    //   //         T.range_count( T.tree.get(), queryBox, 1e-7 );
    //   //         T.range_query( T.tree.get(),
    //   //                        Out.cut( 0, T.tree.get()->get_aug() ),
    //   queryBox,
    //   //                        1e-7 );
    //   //         //  std::cout << T.tree.get()->get_aug() << std::endl;
    //   //       }
    //   //     },
    //   //     [&]() {} );
    //   std::cout << "-1 -1 -1 " << std::flush;
    // }

    if( queryType & ( 1 << 4 ) ) {  //* batch insertion with fraction
      double ratios[10] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
      for( int i = 0; i < 10; i++ ) {
        size_t sz = size_t( vin.size() * ratios[i] );

        double aveInsert = time_loop(
            rounds, 1.0,
            [&]() {
              T.tree.reset();
              v2 = parlay::tabulate(
                  n, [&]( size_t i ) -> vtx* { return &v[i]; } );
              vin2 = parlay::tabulate(
                  sz, [&]( size_t i ) -> vtx* { return &vin[i]; } );
              allv = parlay::append( v2, vin2 );
              whole_box = knn_tree::o_tree::get_box( allv );
              T = knn_tree( v2, whole_box );
            },
            [&]() {
              dims = vin2[0]->pt.dimension();
              root = T.tree.get();
              bd = T.get_box_delta( dims );

              T.batch_insert( vin2, root, bd.first, bd.second );
            },
            [&]() { T.tree.reset(); } );

        std::cout << aveInsert << " " << std::flush;
      }
    }

    if( queryType & ( 1 << 5 ) ) {  //* batch deletion with fraction
      double ratios[10] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
      for( int i = 0; i < 10; i++ ) {
        size_t sz = size_t( v.size() * ratios[i] );
        if( i == 9 ) sz = v.size() - 100;

        double aveDelete = time_loop(
            rounds, 1.0,
            [&]() {
              T.tree.reset();
              v2 = parlay::tabulate(
                  n, [&]( size_t i ) -> vtx* { return &v[i]; } );
              vin2 = parlay::tabulate(
                  sz, [&]( size_t i ) -> vtx* { return &vin[i]; } );
              allv = parlay::append( v2, vin2 );
              whole_box = knn_tree::o_tree::get_box( allv );
              //* warning not the same as others
              T = knn_tree( allv, whole_box );
              dims = vin2[0]->pt.dimension();
              root = T.tree.get();
              bd = T.get_box_delta( dims );
            },
            [&]() {
              vin2 = parlay::tabulate(
                  sz, [&]( size_t i ) -> vtx* { return &vin[i]; } );
              T.batch_delete( vin2, root, bd.first, bd.second );
            },
            [&]() { T.tree.reset(); } );
        std::cout << aveDelete << " " << std::flush;
      }
    }

    if( queryType & ( 1 << 8 ) ) {  //* batch insertion then knn
      //* first run general build
      T.tree.reset();
      v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
      whole_box = knn_tree::o_tree::get_box( v2 );
      T = knn_tree( v2, whole_box );

      auto aveQuery = time_loop(
          rounds, 1.0, [&]() {},
          [&]() {
            k = 100;
            if( algorithm_version == 0 ) {  // this is for starting from
              parlay::sequence<vtx*> vr = T.vertices();
              // find nearest k neighbors for each point
              //  vr = T.vertices();
              //  vr = parlay::random_shuffle( vr.cut( 0, vr.size() ) );
              size_t n = vr.size();
              parlay::parallel_for(
                  0, n, [&]( size_t i ) { T.k_nearest( vr[i], k ); } );
            } else if( algorithm_version == 1 ) {
            } else {
            }
          },
          [&]() {} );
      std::cout << aveQuery << " -1 " << std::flush;

      //* then incremental build
      T.tree.reset();
      v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
      vin2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
      allv = parlay::append( v2, vin2 );
      whole_box = knn_tree::o_tree::get_box( allv );

      size_t sz = n * 0.1;
      vin2.resize( sz );
      parlay::parallel_for( 0, sz, [&]( size_t i ) { vin2[i] = &v[i]; } );
      T = knn_tree( vin2, whole_box );

      size_t l = sz, r = 0;
      while( l < n ) {
        r = std::min( l + sz, n );
        dims = vin2[0]->pt.dimension();
        root = T.tree.get();
        bd = T.get_box_delta( dims );
        vin2.resize( r - l );
        parlay::parallel_for( 0, r - l,
                              [&]( size_t i ) { vin2[i] = &vin[l + i]; } );
        T.batch_insert( vin2, root, bd.first, bd.second );

        l = r;
      }

      aveQuery = time_loop(
          rounds, 1.0, [&]() {},
          [&]() {
            k = 100;
            if( algorithm_version == 0 ) {  // this is for starting from
              parlay::sequence<vtx*> vr = T.vertices();
              // find nearest k neighbors for each point
              //  vr = T.vertices();
              //  vr = parlay::random_shuffle( vr.cut( 0, vr.size() ) );
              size_t n = vr.size();
              parlay::parallel_for(
                  0, n, [&]( size_t i ) { T.k_nearest( vr[i], k ); } );
            } else if( algorithm_version == 1 ) {
            } else {
            }
          },
          [&]() {} );
      std::cout << aveQuery << " -1 " << std::flush;
    }

    if( queryType & ( 1 << 9 ) ) {  //* batch deletion then knn
                                    //* first run general build
      T.tree.reset();
      v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
      whole_box = knn_tree::o_tree::get_box( v2 );
      T = knn_tree( v2, whole_box );

      auto aveQuery = time_loop(
          rounds, 1.0, [&]() {},
          [&]() {
            k = 100;
            if( algorithm_version == 0 ) {  // this is for starting from
              parlay::sequence<vtx*> vr = T.vertices();
              // find nearest k neighbors for each point
              //  vr = T.vertices();
              //  vr = parlay::random_shuffle( vr.cut( 0, vr.size() ) );
              size_t n = vr.size();
              parlay::parallel_for(
                  0, n, [&]( size_t i ) { T.k_nearest( vr[i], k ); } );
            } else if( algorithm_version == 1 ) {
            } else {
            }
          },
          [&]() {} );
      std::cout << aveQuery << " -1 " << std::flush;

      //* then incremental delete
      T.tree.reset();
      v2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &v[i]; } );
      vin2 = parlay::tabulate( n, [&]( size_t i ) -> vtx* { return &vin[i]; } );
      allv = parlay::append( v2, vin2 );
      whole_box = knn_tree::o_tree::get_box( allv );
      T = knn_tree( allv, whole_box );

      size_t sz = n * 0.1;

      size_t l = sz, r = 0;
      while( l < n - 100 ) {
        r = std::min( l + sz, n - 100 );
        dims = vin2[0]->pt.dimension();
        root = T.tree.get();
        bd = T.get_box_delta( dims );
        vin2.resize( r - l );
        parlay::parallel_for( 0, r - l,
                              [&]( size_t i ) { vin2[i] = &vin[l + i]; } );
        T.batch_delete( vin2, root, bd.first, bd.second );
        l = r;
      }

      aveQuery = time_loop(
          rounds, 1.0, [&]() {},
          [&]() {
            k = 100;
            if( algorithm_version == 0 ) {  // this is for starting from
              parlay::sequence<vtx*> vr = T.vertices();
              // find nearest k neighbors for each point
              //  vr = T.vertices();
              //  vr = parlay::random_shuffle( vr.cut( 0, vr.size() ) );
              size_t n = vr.size();
              parlay::parallel_for(
                  0, n, [&]( size_t i ) { T.k_nearest( vr[i], k ); } );
            } else if( algorithm_version == 1 ) {
            } else {
            }
          },
          [&]() {} );
      std::cout << aveQuery << " -1 " << std::flush;
    }

    std::cout << std::endl << std::flush;

    // // t.next( "try all" );
    // // if( report_stats ) {
    // //   auto s = parlay::delayed_seq<size_t>(
    // //       v.size(), [&]( size_t i ) { return v[i]->counter; } );
    // //   size_t i = parlay::max_element( s ) - s.begin();
    // //   size_t sum = parlay::reduce( s );
    // //   std::cout << "max internal = " << s[i]
    // //             << ", average internal = " << sum / ( (double)v.size()
    // )
    // //             << std::endl;
    // //   // t.next( "stats" );
    // // }
    // // t.next( "delete tree" );
  };
}
