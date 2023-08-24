import copy
import io
import json
import logging
import math
import os
import pickle
import random
import time

import geopandas as gpd
import networkx as nx
import pandas as pd
import statsmodels.formula.api as smf
from gerrychain import Graph

"""

The code defines three functions: produce_block_level_graph, partition_blocks_by_vtds, and generate_block_level_graph.

produce_block_level_graph takes in no arguments and produces a block-level graph using the input shapefiles and configuration. It does this by calling partition_blocks_by_vtds to preprocess the input shapefiles and compute the necessary data, and then calling generate_block_level_graph to create the actual graph using the preprocessed data.

partition_blocks_by_vtds takes in several arguments representing the input shapefiles and configuration, and partitions the blocks by VTDs. It returns the processed block data with population data and voting data reapportioned from the VTD to block level.

generate_block_level_graph takes in a GeoDataFrame containing the processed block data, and generates a block-level graph using this data. It returns the generated graph.

Data:

Locations of the shapefiles and population data for the state.

Shapefiles are from the US Census Bureau, Tigerline.

Population data is from: https://www.fcc.gov/economics-analytics/industry-analysis-division/staff-block-estimates

Todo: Are these the right data sources for the state?

We can also use block groups and then get demographic data from the ACS, but this is not implemented yet.


"""


def main():
    """
    Produces a block-level graph using the input shapefiles and configuration.

    Returns:
        graph (nx.Graph): The generated block-level graph.
    """

    config = {
        "VTD_shp_file": os.path.join(os.path.dirname(__file__), "input/NC_VTD.shp"),
        "sub_partition_shape_file": os.path.join(
            os.path.dirname(__file__), "input/tl_2016_37_tabblock10.shp"
        ),
        "sub_partition_block_id": "GEOID10",
        "sub_partition_pop_data": "./input/us2016.csv",
        "sub_partition_block_id_pop_table": "block_fips",
    }

    gdf_blocks = partition_blocks_by_vtds(**config)
    graph = generate_block_level_graph(gdf_blocks)

    return graph


def impute_down(
    coarse_partition="VTD",
    coarse_id="vtdid",
    coard_pop_field="POP",
    coarse_field_to_impute="VOTE",
    fine_partition="BLOCKS",
    fine_id="blockid",
    fine_pop_field="POP",
    fine_pop_demographics=None,
):
    """
    Imputes values from a coarse partition to a fine partition.

    Args:
        coarse_partition (GeoDataFrame): The coarse partition.
        coarse_id (str): The ID field for the coarse partition.
        coarse_pop_field (str) : The population field for the coarse partition.
        coarse_field_to_impute (str): The field to impute from the coarse partition.
        fine_partition (GeoDataFrame): The fine partition.
        fine_id (str): The ID field for the fine partition.
        fine_pop_field (str) : The population field for the fine partition.

    Returns:
        fine_partition (GeoDataFrame): The fine partition with imputed values.
    """

    joined = gpd.sjoin(
        coarse_partition, fine_partition, how="left", predicate="intersects"
    )
    mapping = joined[[coarse_id, fine_id]]

    fine_with_map = fine_partition.merge(mapping)

    joined_with_geos = fine_with_map.merge(
        coarse_partition,
        left_on=coarse_id,
        right_on=coarse_id,
        suffixes=("_fine", "_coarse"),
    )

    fine_geos = gpd.GeoSeries(joined_with_geos.geometry_fine).to_crs("ESRI:103500")
    joined_with_geos["geo_intersect"] = fine_geos.intersection(
        gpd.GeoSeries(joined_with_geos.geometry_coarse).to_crs("ESRI:103500")
    )
    joined_with_geos["geo_fine_area"] = fine_geos.area
    joined_with_geos["intersect_area"] = (
        gpd.GeoSeries(joined_with_geos.geo_intersect).to_crs("ESRI:103500").area
    )
    joined_with_geos["area_proportion"] = (
        joined_with_geos["intersect_area"] / joined_with_geos["geo_fine_area"]
    )

    joined_with_geos_filtered = joined_with_geos[
        joined_with_geos.area_proportion >= 0.99
    ]

    if len(joined_with_geos_filtered) / len(fine_partition) < 0.5:
        logging.warning("Lost most of the fine partition")


def partition_blocks_by_vtds(
    VTD_shp_file,
    sub_partition_shape_file,
    sub_partition_block_id="GEOID10",
    sub_partition_pop_data="./data/us2016.csv",
    sub_partition_block_id_pop_table="block_fips",
):
    """
    Partitions blocks by VTDs and computes necessary data.

    Args:
        VTD_shp_file (str): Path to the shapefile for VTDs.
        sub_partition_shape_file (str): Path to the shapefile for blocks.
        sub_partition_block_id (str): The name of the column that contains the block IDs.
        sub_partition_pop_data (str): Path to the population data for blocks.
        sub_partition_block_id_pop_table (str): The name of the column that contains the block IDs in the population data.

    Returns:
        geo_with_pop_data (gpd.GeoDataFrame): The processed block data with population data.
    """

    # file to check for existence
    filename = os.path.join(
        os.path.dirname(__file__), "processed_data_outputs/blocks_with_votes.shp"
    )

    # check if file exists
    if os.path.exists(filename):
        print("File already exists")
        gdf_dedupped = gpd.read_file(filename)
        return gdf_dedupped

    vtd = gpd.read_file(VTD_shp_file).to_crs("EPSG:4269")
    blocks = gpd.read_file(sub_partition_shape_file).to_crs("EPSG:4269")

    joined = gpd.sjoin(blocks, vtd, how="left", predicate="intersects")
    mapping = joined[[sub_partition_block_id, "VTD"]]

    # Intersect geos with VTDs, compute area ratios

    # add back in block shape data
    geo_table_with_shape_data = blocks.merge(mapping)
    # add back in VTD shape data
    geo_with_vtd_shape_data = geo_table_with_shape_data.merge(
        vtd, left_on="VTD", right_on="VTD", suffixes=("_block", "_vtd")
    )

    geo_geos = gpd.GeoSeries(geo_with_vtd_shape_data.geometry_block).to_crs(
        "ESRI:103500"
    )
    geo_with_vtd_shape_data["geo_intersect"] = geo_geos.intersection(
        gpd.GeoSeries(geo_with_vtd_shape_data.geometry_vtd).to_crs("ESRI:103500")
    )
    geo_with_vtd_shape_data["geo_area"] = geo_geos.area
    geo_with_vtd_shape_data["intersect_area"] = (
        gpd.GeoSeries(geo_with_vtd_shape_data.geo_intersect).to_crs("ESRI:103500").area
    )
    geo_with_vtd_shape_data["area_proportion"] = (
        geo_with_vtd_shape_data["intersect_area"] / geo_with_vtd_shape_data["geo_area"]
    )

    ## Keep only subpartition blocks that mostly intersect;
    size_before = len(geo_with_vtd_shape_data)
    geo_with_vtd_shape_data = geo_with_vtd_shape_data[
        geo_with_vtd_shape_data.area_proportion >= 0.99
    ]

    if len(geo_with_vtd_shape_data) / size_before < 0.5:
        logging.warning("Lost most of the subpartition bocks")
    # Impute votes
    # Later replace this with EI.

    pops = pd.read_csv(sub_partition_pop_data)
    pops["block_fips"] = pops.block_fips.astype("str")
    vote_fields = ["EL16G_GV_R", "EL16G_GV_D", "TOTPOP"]
    fields = [
        "VTD",
        sub_partition_block_id,
        "geometry",
        "area_proportion",
    ] + vote_fields
    geos_simple = geo_with_vtd_shape_data.rename(
        columns={"geometry_block": "geometry"}
    )[fields]
    geo_with_pop = geos_simple.merge(
        pops,
        left_on=sub_partition_block_id,
        right_on=sub_partition_block_id_pop_table,
    )

    # VTD level sums --
    # ###  doing because the VTD tot pop is generally not equal to the sum of the block pops in it
    # This indicates some data level issue , probably in addition to the non exact containment issue.
    # Code to check:
    # merged = geo_with_pop[["TOTPOP", "VTD"]].merge(geo_with_pop[["pop2016", "VTD"]].groupby("VTD").agg(sum).reset_index(), left_on = "VTD", right_on = "VTD")
    # (merged.TOTPOP / merged.pop2016).sort_values().apply(lambda x : round(x,1) ).value_counts()
    # But ignoring for now.
    vtds_block_pops = geo_with_pop[["pop2016", "VTD"]].groupby("VTD").agg(sum)
    vtds_block_pops.columns = ["pop2016_VTD_total"]
    geo_with_pop = geo_with_pop.merge(vtds_block_pops, left_on="VTD", right_index=True)
    geo_with_pop["pop_proportion"] = (
        geo_with_pop.pop2016 / geo_with_pop.pop2016_VTD_total
    )

    vote_fields = ["EL16G_GV_R", "EL16G_GV_D", "TOTPOP"]
    for field in vote_fields:
        # Eventually replace with ecological inference or something
        geo_with_pop[field + "_block"] = (
            geo_with_pop["pop_proportion"] * geo_with_pop[field]
        )

    gdf_dedupped = geo_with_pop.sort_values("area_proportion").drop_duplicates(
        "GEOID10", keep="last"
    )
    gdf_dedupped.pop2016.sum() / geo_with_pop.pop2016.sum()

    gdf_subset = gdf_dedupped[
        [
            "VTD",
            "GEOID10",
            "geometry",
            "TOTPOP",
            "pop2016",
            "pop_proportion",
            "area_proportion",
        ]
        + [x + "_block" for x in vote_fields]
    ]
    gdf_subset = gpd.GeoDataFrame(gdf_subset)

    gdf_subset = gpd.GeoDataFrame(
        gdf_subset, geometry="geometry", crs="ESRI:103500"
    )  # .head(10)
    gdf_subset.to_file("./processed_data_outputs/blocks_with_votes.shp")

    return gdf_subset


def generate_block_level_graph(blocks=None):
    """
    Generates a block-level graph using the input block data.

    Args:
        gdf_blocks (gpd.GeoDataFrame): The processed block data.

    Returns:
        graph (nx.Graph): The generated block-level graph.
    """
    # file to check for existence
    filename = "./processed_data_outputs/blocks_graph.json"

    # check if file exists
    if os.path.exists(filename):
        print("Graph Already Exists")
        blocks_graph = Graph.from_json(filename)
        return blocks_graph

    if blocks is None:
        blocks = gpd.read_file("./processed_data_outputs/blocks_with_votes.shp").to_crs(
            "EPSG:4269"
        )

    # Generate graph

    blocks["pos"] = blocks.centroid
    blocks = blocks.rename(columns={"EL16G_GV_D": "dem", "EL16G_GV_R": "rep"})
    blocks["dem_prop"] = blocks.dem / blocks.TOTPOP
    blocks["rep_prop"] = blocks.rep / blocks.TOTPOP

    blocks_graph = Graph.from_geodataframe(blocks, ignore_errors=True)
    blocks_graph.to_json(
        "./processed_data_outputs/blocks_graph.json", include_geometries_as_geojson=True
    )

    return blocks_graph


def condensce_by_partition(graph, part):
    """
    Condenses a graph by partition,
    creating a new graph where each node represents a block and each edge represents a connection between two blocks.

    Also, adds up the population and votes in each block.

    Args:
        graph (nx.Graph): The input graph.
        partitions (dict): A gerrychain partitino object.

    Returns:
        condensed_graph (nx.Graph): The condensed graph.
    """
    graph_copy = graph.copy()

    for supernode in part.parts.values():
        nodes = list(supernode)
        for node in nodes[1:]:
            tot_pop = (
                graph_copy.nodes[nodes[0]]["TOTPOP_block"]
                + graph_copy.nodes[node]["TOTPOP_block"]
            )
            tot_dem = graph_copy.nodes[nodes[0]]["dem"] + graph_copy.nodes[node]["dem"]
            tot_rep = graph_copy.nodes[nodes[0]]["rep"] + graph_copy.nodes[node]["rep"]
            nx.contracted_nodes(graph_copy, nodes[0], node, copy=False)
            graph_copy.nodes[nodes[0]]["TOTPOP_block"] = tot_pop
            graph_copy.nodes[nodes[0]]["dem"] = tot_dem
            graph_copy.nodes[nodes[0]]["rep"] = tot_rep

    return graph_copy


if __name__ == "__main__":
    # Running
    main()
