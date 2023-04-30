TODO: where did NC.py come from?

TX.json came from TX_vtds.shp (here: https://github.com/mggg-states/TX-shapefiles),
and then converted to JSON using gerrychain:
import geopandas as gp
from gerrychain import Graph
g = gp.read_file('TX_vtds.dbf')
g['pos'] = g.centroid
Graph.from_geodataframe(g, ignore_errors=True).to_json('TX.json', include_geometries_as_geojson=True)