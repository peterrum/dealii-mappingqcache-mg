// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Test MappingQCache initialization with point lambda/Function

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <deal.II/numerics/data_out.h>

using namespace dealii;

class ScopedTimer
{
public:
  ScopedTimer(double &result)
    : result(result)
    , temp(std::chrono::system_clock::now())
  {}

  ~ScopedTimer()
  {
    result += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now() - temp)
                .count() /
              1e9;
  }

private:
  double &                                           result;
  std::chrono::time_point<std::chrono::system_clock> temp;
};

template <int dim>
void
do_test(const unsigned int degree,
        const unsigned int mapping_degree,
        const bool         is_displacement_function)
{
  double time_0 = 0.0;
  double time_1 = 0.0;

  const unsigned int n_refinements = 4;

  // version 1: initialize MappingQCache on the levels directly based on a
  // solution vector and using RemotePointEvaluation
  {
    ScopedTimer timer(time_0);

    parallel::distributed::Triangulation<dim> tria(
      MPI_COMM_WORLD,
      Triangulation<dim>::MeshSmoothing::none,
      parallel::distributed::Triangulation<
        dim>::Settings::construct_multigrid_hierarchy);
    GridGenerator::subdivided_hyper_cube(tria, 1);
    tria.refine_global(n_refinements);

    FESystem<dim>   fe(FE_Q<dim>(degree), dim);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    LinearAlgebra::distributed::Vector<double> vector(
      dof_handler.locally_owned_dofs(),
      locally_relevant_dofs,
      dof_handler.get_communicator());

    MappingQGeneric<dim> mapping(mapping_degree);
    MappingQCache<dim>   mapping_cache(mapping_degree);
    mapping_cache.initialize(mapping,
                             dof_handler,
                             vector,
                             is_displacement_function);
  }

  // version 2: initialize MappingQCache on the levels in two steps: interpolate
  // the result onto the multigrid levels and use these vectors for setup
  {
    ScopedTimer                               timer(time_1);
    parallel::distributed::Triangulation<dim> tria(
      MPI_COMM_WORLD,
      Triangulation<dim>::MeshSmoothing::none,
      parallel::distributed::Triangulation<
        dim>::Settings::construct_multigrid_hierarchy);
    GridGenerator::subdivided_hyper_cube(tria, 1);
    tria.refine_global(n_refinements);

    FESystem<dim>   fe(FE_Q<dim>(degree), dim);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    LinearAlgebra::distributed::Vector<double> vector(
      dof_handler.locally_owned_dofs(),
      locally_relevant_dofs,
      dof_handler.get_communicator());

    MGLevelObject<LinearAlgebra::distributed::Vector<double>> vectors(
      0, tria.n_global_levels() - 1);
    MGTransferMatrixFree<dim, double> transfer;
    transfer.build(dof_handler);
    transfer.interpolate_to_mg(dof_handler, vectors, vector);

    MappingQGeneric<dim> mapping(mapping_degree);
    MappingQCache<dim>   mapping_cache(mapping_degree);
    mapping_cache.initialize(mapping,
                             dof_handler,
                             vectors,
                             is_displacement_function);
  }

  // print timings
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << time_0 << " " << time_1 << std::endl;
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  do_test<3>(4, 4, true);
}
