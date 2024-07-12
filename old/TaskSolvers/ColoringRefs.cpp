#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdbool.h>
#include <random>


//Actual functions --------------------------------------------------------
/************************************************************************/

//recursive function - for internal module use only
// void nofunc(void){}


bool search_recursion(int *edges, int *coloring, int nnodes, int ncolors, int node_indx){

  if (nnodes == node_indx) {return true;}

  for (int color = 0; color<ncolors; color++){
    coloring[node_indx] = color;

    bool ok = true;
    for (int j=0; j<nnodes; j++){
      if (edges[node_indx*nnodes + j] == 1 && coloring[j] == color) {ok = false;} 
    }
    if (ok) {
      if (search_recursion(edges, coloring, nnodes, ncolors, node_indx+1)) {return true;}
    }

  }
  coloring[node_indx] = -1;
  return false;
}


PyObject *ColoringNonRedundantSearch(PyObject *self, PyObject *args) {
  import_array(); //numpy initialization function, otherwise may get seg faults
  int nColors;
  PyArrayObject *edges_np;

  // Parse arguments. 
  if (!PyArg_ParseTuple(args, "iO",
                        &nColors, 
                        &edges_np)) {
    return NULL;
  }

  int nNodes = PyArray_SHAPE(edges_np)[0];
  int *edges = (int*)PyArray_DATA(edges_np);

  //initialize coloring to indicate that it is not colored:
  int *coloring = (int *)malloc(nNodes*sizeof(int));
  for (int i = 0; i<nNodes; i++){
    coloring[i] = -1;
  }

  //first node is set to 0 by default - no need to iterate through colors of the first node
  coloring[0] = 0;

  //algorithm meat is in recursion function.
  //Fills values into the coloring vector
  if (!search_recursion(edges, coloring, nNodes, nColors, 1))
    coloring[0] = -1; //set to indicate failure


  //wrap best state in a numpy structure and send it back to python:
  npy_intp const dims [1] = {nNodes};
  PyObject* array3 = PyArray_SimpleNewFromData(1, dims, NPY_INT, coloring);
  return array3;

};

int calc_move_de(int *node_edges, int *coloring, int node2try, int color2try){
  //the node_edges pointer should already point to the start of the correct row of the larger edges matrix

  int current_color = coloring[node2try];
  if (current_color == color2try) {
    //no change, so we already know what the dE is:
    return 0;
  }else{
    int current_bad = 0;
    int proposed_bad = 0;
    int degree_lim = node_edges[0];
    
    for (int i = 1; i<degree_lim; i++){
      int connected_node = node_edges[i];
      int conn_node_color = coloring[connected_node];
      if (conn_node_color == current_color) {current_bad++;} 
      if (conn_node_color == color2try) {proposed_bad++;} 
    }
    return proposed_bad - current_bad;
  }
}


PyObject *ColoringStreamlinedPotts(PyObject *self, PyObject *args) {
  //essentially runs the Potts algorithm, but on a streamlined datastructure tailored to graph coloring

  import_array(); //numpy initialization function, otherwise may get seg faults

  //create random number generator:
  std::mt19937_64 generator;
  std::uniform_real_distribution<float> distribution(0.0,1.0);

  int nColors, maxiters;
  float temp;
  PyArrayObject *edges_np;

  // Parse arguments. 
  if (!PyArg_ParseTuple(args, "iifO",
                        &nColors, 
                        &maxiters,
                        &temp,
                        &edges_np)) {
    return NULL;
  }

  int nNodes = PyArray_SHAPE(edges_np)[0];
  int maxDeg = PyArray_SHAPE(edges_np)[1];
  int* edges = (int*)PyArray_DATA(edges_np);


  //initialize coloring to a uniform coloring
  int *coloring = (int *)malloc(nNodes*sizeof(int));
  for (int i = 0; i<nNodes; i++){
    coloring[i] = 0;
  }


  //keep track of how many bad edges there are.  We will incrimentally adjust this number rather than recalculate
  int bad_edges = 0;
  for (int i = 0; i<nNodes; i++){
    //since initial coloring is uniform, we only need to count the edges;
    //number of edges on each node is stored in the first index, so just add:
    bad_edges = bad_edges + edges[i*maxDeg]-1;
  }
  bad_edges = bad_edges / 2; //except that the above loop double counts.  Easy to fix!

  for (int iter = 0; iter < maxiters; iter++){
    //randomly choose a node and color to attempt an update:
    int node2try = rand()%nNodes;
    int color2try = rand()%nColors;

    int current_color = coloring[node2try];
    if (current_color == color2try) {continue;} //no change is happening no way no how, so just skip thru to the end

    //calculate difference in bad edges if update is accepted:
    int * node_edges = edges+node2try*maxDeg;
    float move_de = calc_move_de(node_edges, coloring, node2try, color2try);

    //decide wether or not to accept the update:
    float p_move;
    if (move_de <= 0) p_move = 1;
    else p_move = exp(-move_de/temp);
    float rnd = distribution(generator);


    if (p_move > rnd) {
      bad_edges = bad_edges + move_de;
      coloring[node2try] = color2try;
      if (bad_edges <= 0){
          break;
      }
    }

  }

  npy_intp const dims [1] = {nNodes};
  PyObject* array3 = PyArray_SimpleNewFromData(1, dims, NPY_INT, coloring);
  return array3;

  // Py_RETURN_NONE;
}