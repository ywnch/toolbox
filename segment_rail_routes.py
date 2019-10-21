def download_rail_route(rr_file, verbose=False):
    """
    Downloads rail route data from railrouter.sg repo.
    Source: https://github.com/cheeaun/railrouter-sg

    params
    ------
    rr_file (str): filename in the data directory in the repo
    verbose (bool): print downloaded route and pattern info

    returns
    -------
    rr_data (dict): the core component from the raw data
    patterns (list): a list of patterns
    """
    import json
    import requests

    url = 'https://raw.githubusercontent.com/' \
          'cheeaun/railrouter-sg/master/data/v2/'
    rr_data = json.loads(requests.get(url + rr_file).text)['routes'][0]
    patterns = rr_data['patterns']

    if verbose:
        print("[Downloaded]")
        print("Route:", rr_data['name'])
        print("Patterns:")
        print('\n'.join(
            [(str(i) + ': ' + p['name']) for i, p in enumerate(patterns)]))

    return rr_data, patterns


def concat_sttn_name(x):
    """
    Customized helper to make sure station names have unified format:
    LikeThisOne.
    """
    if len(x) == 1:
        w = x[0]
        return w[0].upper() + w[1:]
    else:
        return ''.join([w.capitalize() for w in x])


def get_segmentation(pattern):
    """
    Performs route segmentation of a given pattern and return as a GeoDataFrame.

    params
    ------
    pattern (dict): one of the patterns from download_rail_route()

    returns
    -------
    gdf (GeoDataFrame): containing each consecutive station pair as a row
    path (list): sequential coords [lon, lat] that compose the pattern shape
    sttn (list): sequential coords [lon, lat] of all stations along the pattern
    lines (list): sequential shapely.geometry.LineString for each segmented link

    notes
    -----
    - The coordinates system (crs) of gdf is in epsg: 3857 (meter as units,
    instead of lon, lat).
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    path = [[x, y] for y, x in pattern['path']]  # reverse lat, lon order
    sttn_name = [concat_sttn_name(s['id'].split('_')[1:])
                 for s in pattern['stop_points']]
    sttn_idx = [s['path_index'] for s in pattern['stop_points']]  # starts 0
    sttn = [path[i] for i in sttn_idx]

    name_pairs = list(zip(sttn_name[:-1], sttn_name[1:]))
    idx_pairs = list(zip(sttn_idx[:-1], sttn_idx[1:]))
    lines = [LineString(path[o:d + 1]) for o, d in idx_pairs]

    gdf = gpd.GeoDataFrame({'station_pair': name_pairs}, geometry=lines)
    gdf.crs = {'init': 'epsg:4326'}
    gdf = gdf.to_crs(epsg=3857)  # calculate length in meters
    gdf['length'] = gdf['geometry'].length
    gdf = gdf.to_crs(epsg=4326)  # convert back to (lon, lat)

    return gdf, path, sttn, lines


def get_all_links(rr_files):
    """
    Do segmentation for all routes and patterns from a list of rr_files.

    returns
    -------
    rr_files (list): list of json file names containing route data
    links (DataFrame): a full DataFrame of all station links, not unique

    notes
    -----
    - The results can contain multiple entries as there may be different tracks for the same link.
    """
    import pandas as pd
    gdfs = []

    for rr_file in rr_files:
        rr_data, patterns = download_rail_route(rr_file, verbose=False)
        for pattern in patterns:
            gdf, *_ = get_segmentation(pattern)
            gdfs.append(gdf)

    links = pd.concat(gdfs, ignore_index=True)
    return links


def get_output(links_df, out_path=None):
    """
    Produces the final output with unique rail route links and respective length and geometry.
    links_df (DataFrame-like): as generated from get_all_links() or get_segmentation()
    """
    import numpy as np
    import geopandas as gpd

    # decide to use shortest or longest track for the same link
    links = links_df.sort_values('length', ascending=False)
    # unify the station name pair order per link
    links['station_pair'] = links['station_pair'].apply(
        lambda x: tuple(np.sort(x)))
    # reduce to single length per link
    links = links.groupby('station_pair').first().reset_index()
    # create geometry
    links = gpd.GeoDataFrame(links, geometry='geometry')
    # convert dtype inorder to save as json
    links['station_pair'] = links['station_pair'].apply(lambda x: '<>'.join(x))
    # save output
    if out_path is not None:
        links.to_file(out_path + 'sg_rail_links.json', driver='GeoJSON')
    return links


def plot_segmentation(path, sttn, lines, roadnet=None, out_path=None):
    """
    Plots the segmentation output with previously generated files.

    params
    ------
    roadnet (GeoDataFrame): download with load_roadnet()
    out_path (str): directory for saving the figure, if not None

    notes
    -----
    - Roadnet is used to plot the basemap and scale the pattern properly, but is
    not required.
    """
    import matplotlib.pyplot as plt

    # get x, y for plotting
    path_x, path_y = list(zip(*path))
    sttn_x, sttn_y = list(zip(*sttn))

    # plot basemap if provided
    if roadnet is not None:
        roadnet.plot(figsize=(28, 18), linewidth=0.1, color='gray')

    # plot original route pattern and stations
    plt.plot(path_x, path_y, linewidth=10, color='steelblue', alpha=0.3)
    plt.scatter(sttn_x, sttn_y, s=50, c='indianred', alpha=1)

    # plot segmented links
    for l in lines:
        plt.plot(l.xy[0], l.xy[1], linewidth=3)

    # save figure if path is specified
    if out_path is not None:
        plt.title("Rail Route Link Segmentation Output", fontsize=26)
        plt.savefig(out_path + 'link_segmentation.png',
                    bbox_inches='tight', dpi=250)


def load_roadnet():
    """
    Loads road network graph edges of Singapore as a GeoDataFrame.
    The data is downloaded from OpenStreetMap through osmnx package.
    run `!pip install osmnx` if not yet installed
    documentation: https://github.com/gboeing/osmnx

    returns
    -------
    roadnet (GeoDataFrame): road network graph edges of Singapore
    """
    import osmnx as ox

    # download from OpenStreetMap if local datasets is not available
    print("Trying to download the network of Singapore through OSM Nominatim.")

    # try different query results
    n = 1
    while n <= 5:
        try:
            G = ox.graph_from_place(query='Singapore', network_type='drive',
                                    which_result=n)
            break
        except:
            n += 1

    roadnet = ox.graph_to_gdfs(G, nodes=False, edges=True)
    print("Singapore roadnet is downloaded and loaded as a GeoDataFrame.")

    return roadnet
