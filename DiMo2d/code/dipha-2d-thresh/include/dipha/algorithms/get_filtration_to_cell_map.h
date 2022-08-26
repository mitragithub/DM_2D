/*  Copyright 2014 IST Austria

Contributed by: Jan Reininghaus

This file is part of DIPHA.

DIPHA is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DIPHA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with DIPHA.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <dipha/includes.h>

namespace dipha
{
  namespace algorithms
  {
    template< typename Complex >
    void get_filtration_to_cell_map(const inputs::abstract_weighted_cell_complex< Complex >& complex,
                                    bool dualize,
                                    data_structures::distributed_vector< int64_t >& filtration_to_cell_map)
    {
      const int64_t global_num_cells = complex.get_num_cells();

      std::cout << "Number of Cells: " << global_num_cells << std::endl;

      const int64_t local_begin = element_distribution::get_local_begin(global_num_cells);
      const int64_t local_end = element_distribution::get_local_end(global_num_cells);
      const int64_t local_num_cells = local_end - local_begin;

      std::cout << local_begin << " " << local_end << std::endl;

      typedef std::pair< double, std::pair< int64_t, int64_t > > sort_value_type;
      //std::vector< sort_value_type > filtration(global_num_cells);
      std::vector< sort_value_type > filtration;
      int64_t tracking = 0;
      for (int64_t cur_cell = local_begin; cur_cell < local_end; cur_cell++)
      {
        /*
        if (cur_cell % 1000000 == 0)
        {
          std::cout << " adding value: " << complex.get_local_value(cur_cell) << std::endl;
        }
        */
        if (complex.get_local_value(cur_cell) >= 224)
        {
          continue;
        }
        tracking++;
        
        filtration.push_back(std::make_pair(complex.get_local_value(cur_cell),
           std::make_pair(complex.get_local_dim(cur_cell), cur_cell)));
        
        /*
        filtration[cur_cell - local_begin].first = complex.get_local_value(cur_cell);
        filtration[cur_cell - local_begin].second.first = complex.get_local_dim(cur_cell);
        filtration[cur_cell - local_begin].second.second = cur_cell;
        */
      }

      std::cout << "filtration: " << tracking << " / " << global_num_cells << std::endl;

      // psort unfortunately uses long for the size. This will cause problems on Win64 for large data
      
      std::vector< long > cell_distribution;
      int num_processes = mpi_utils::get_num_processes();
      for (int cur_rank = 0; cur_rank < num_processes; cur_rank++)
        cell_distribution.push_back((long)(element_distribution::get_local_end(global_num_cells, cur_rank) 
                                           - element_distribution::get_local_begin(global_num_cells, cur_rank)));

      if (dualize)
        p_sort::parallel_sort(filtration.begin(), filtration.end(), std::greater< sort_value_type >(), 
                              cell_distribution.data(), MPI_COMM_WORLD);
      else
        p_sort::parallel_sort(filtration.begin(), filtration.end(), std::less< sort_value_type >(), 
                              cell_distribution.data(), MPI_COMM_WORLD);

      std::cout << "sorted" << std::endl;

      filtration_to_cell_map.init(filtration.size());
      for (int64_t cur_cell = local_begin; cur_cell < local_begin + filtration.size(); cur_cell++)
      {
        //std::cout << "cur_cell: " << cur_cell << std::endl;
        //std::cout << "value: " << filtration[cur_cell - local_begin].second.second << std::endl;
        filtration_to_cell_map.set_local_value(cur_cell, filtration[cur_cell - local_begin].second.second);
      }

      /*
      filtration_to_cell_map.init(global_num_cells);
      for (int64_t cur_cell = local_begin; cur_cell < local_end; cur_cell++)
      {
        std::cout << "cur_cell: " << cur_cell << std::endl;
        std::cout << "value: " << filtration[cur_cell - local_begin].second.second << std::endl;
        filtration_to_cell_map.set_local_value(cur_cell, filtration[cur_cell - local_begin].second.second);
      }
      */
      std::cout << "next" << std::endl;
    }
  }
}
