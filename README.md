## Installation

The module is not packaged, simply copy the scripts/functions you need to your working directory/notebook and import it locally.

## Toolbox

> A toolbox for geospatial related tasks, but only one tool exists so far.

- `connect_poi()`: integrate a set of POIs onto the road network based on the nearest projected point

## Segment Rail Routes

> Download Singapore's rail network from RailRouter SG and segment the LineString by stations. This gives the user the shapes and (nearly) actual lengths of each rail link between consecutive stations of a route.

- `demo_segment_rail_routes.ipynb`: see instructions inside, the segmentation result is illustrated below. Or, you may also throw [this file](asset/sg_rail_links_viz.json) into [kepler.gl](https://kepler.gl/demo) to visualize and explore the output.

![](asset/rail_link_segmentation.png)
