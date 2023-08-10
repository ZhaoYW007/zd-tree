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

// find the k nearest neighbors for all points in tree
// places pointers to them in the .ngh field of each vertex
template <int max_k, class vtx>
void
ANN( const parlay::sequence<vtx*>& v, int k, int rounds,
     const parlay::sequence<vtx*>& vin, int tag )
{
   //  timer t( "ANN", report_stats );

   {
      timer _t;
      _t.start();
      using knn_tree = k_nearest_neighbors<vtx, max_k>;
      using node = typename knn_tree::node;
      using box = typename knn_tree::box;
      using box_delta = std::pair<box, double>;

      // create sequences for insertion and deletion
      size_t size = v.size();
      // size_t p = .5 * size;
      // parlay::sequence<vtx*> v1 = parlay::sequence<vtx*>( p );
      // parlay::parallel_for(
      //     0, size,
      //     [&]( size_t i )
      //     {
      //        if( i < p )
      //           v1[i] = v[i];
      //        else
      //           v2[i - p] = v[i];
      //     },
      //     1 );

      // build tree with optional box

      parlay::sequence<vtx*> v2 = parlay::sequence<vtx*>( v.size() );
      parlay::copy( v, v2 );

      box whole_box = knn_tree::o_tree::get_box( v2 );
      knn_tree T = knn_tree( v2, whole_box );

      double aveBuild = time_loop(
          rounds, 1.0,
          [&]()
          {
             parlay::copy( v, v2 );
             whole_box = knn_tree::o_tree::get_box( v2 );
          },
          [&]() { T = knn_tree( v2, whole_box ); }, [&]() { T.tree.reset(); } );
      std::cout << aveBuild << " " << std::flush;

      //* restore
      parlay::copy( v, v2 );
      whole_box = knn_tree::o_tree::get_box( v2 );
      T = knn_tree( v2, whole_box );

      // prelims for insert/delete
      int dims;
      node* root;
      box_delta bd;
      parlay::sequence<vtx*> vin2 = parlay::sequence<vtx*>( vin.size() );

      //* batch-dynamic insertion
      if( tag >= 1 )
      {
         double aveInsert = time_loop(
             rounds, 1.0,
             [&]()
             {
                T.tree.reset();
                parlay::copy( v, v2 );
                parlay::copy( vin, vin2 );
                auto allv = parlay::append( v2, vin2 );

                whole_box = knn_tree::o_tree::get_box( allv );
                T = knn_tree( v2, whole_box );

                dims = vin2[0]->pt.dimension();
                root = T.tree.get();
                bd = T.get_box_delta( dims );
             },
             [&]() { T.batch_insert( vin2, root, bd.first, bd.second ); },
             [&]() { T.tree.reset(); } );
         std::cout << aveInsert << " " << std::flush;

         //* restore
         T.tree.reset();
         parlay::copy( v, v2 );
         parlay::copy( vin, vin2 );
         auto allv = parlay::append( v2, vin2 );

         whole_box = knn_tree::o_tree::get_box( allv );
         T = knn_tree( v2, whole_box );

         dims = vin2[0]->pt.dimension();
         root = T.tree.get();
         bd = T.get_box_delta( dims );
         T.batch_insert( vin2, root, bd.first, bd.second );
         //! no need to append vin since KNN graph always get points from the
         //! tree
      }
      else
      {
         std::cout << "-1 " << std::flush;
      }

      if( tag >= 2 )
      {
         double aveDelete = time_loop(
             rounds, 1.0,
             [&]()
             {
                T.tree.reset();
                parlay::copy( v, v2 );
                parlay::copy( vin, vin2 );
                auto allv = parlay::append( v2, vin2 );

                whole_box = knn_tree::o_tree::get_box( allv );
                T = knn_tree( v2, whole_box );

                dims = vin2[0]->pt.dimension();
                root = T.tree.get();
                bd = T.get_box_delta( dims );
                T.batch_insert( vin2, root, bd.first, bd.second );
             },
             [&]() { T.batch_delete( vin2, root, bd.first, bd.second ); },
             [&]() { T.tree.reset(); } );
         std::cout << aveDelete << " " << std::flush;

         T.tree.reset();
         parlay::copy( v, v2 );
         parlay::copy( vin, vin2 );
         auto allv = parlay::append( v2, vin2 );

         whole_box = knn_tree::o_tree::get_box( allv );
         T = knn_tree( v2, whole_box );

         dims = vin2[0]->pt.dimension();
         root = T.tree.get();
         bd = T.get_box_delta( dims );
         T.batch_insert( vin2, root, bd.first, bd.second );
         T.batch_delete( vin2, root, bd.first, bd.second );
      }
      else
      {
         std::cout << "-1 " << std::flush;
      }

      // batch-dynamic deletion
      // T.batch_delete(v2, root, bd.first, bd.second);
      // //t.next("batch deletion");

      // if( report_stats )
      // std::cout << "depth = " << T.tree->depth() << std::endl;
      parlay::sequence<vtx*> vr;
      auto aveQuery = time_loop(
          rounds, 1.0,
          [&]()
          {
             //  vr = T.vertices();
             //  vr = parlay::random_shuffle( vr.cut( 0, vr.size() ) );
          },
          [&]()
          {
             if( algorithm_version == 0 )
             {  // this is for starting from root
                // this reorders the vertices for locality
                // t.next( "flatten tree" );
                //  parlay::sequence<vtx*> vr = T.vertices();

                // find nearest k neighbors for each point
                vr = T.vertices();
                //  vr = parlay::random_shuffle( vr.cut( 0, vr.size() ) );
                size_t n = vr.size();
                parlay::parallel_for(
                    0, n, [&]( size_t i ) { T.k_nearest( vr[i], k ); } );
             }
             else if( algorithm_version == 1 )
             {
                parlay::sequence<vtx*> vr = T.vertices();
                // t.next( "flatten tree" );

                int dims = ( v[0]->pt ).dimension();
                node* root = T.tree.get();
                box_delta bd = T.get_box_delta( dims );
                size_t n = vr.size();
                parlay::parallel_for(
                    0, n,
                    [&]( size_t i )
                    {
                       T.k_nearest_leaf(
                           vr[i],
                           T.find_leaf( vr[i]->pt, root, bd.first, bd.second ),
                           k );
                    } );
             }
             else
             {  //(algorithm_version == 2) this is for starting from leaf,
                // finding leaf
                //  using map()
                auto f = [&]( vtx* p, node* n )
                { return T.k_nearest_leaf( p, n, k ); };

                // find nearest k neighbors for each point
                T.tree->map( f );
             }
          },
          [&]() {} );
      std::cout << aveQuery << " -1 -1" << std::endl << std::flush;

      // t.next( "try all" );
      if( report_stats )
      {
         auto s = parlay::delayed_seq<size_t>(
             v.size(), [&]( size_t i ) { return v[i]->counter; } );
         size_t i = parlay::max_element( s ) - s.begin();
         size_t sum = parlay::reduce( s );
         std::cout << "max internal = " << s[i]
                   << ", average internal = " << sum / ( (double)v.size() )
                   << std::endl;
         // t.next( "stats" );
      }
      // t.next( "delete tree" );
   };
}
