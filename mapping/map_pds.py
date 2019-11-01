import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os

##objects = list of item; [(label, (long, lat))]

def map_pds(objects):
	os.environ['PROJ_LIB']=r"C:\Users\rishishah\anaconda3\Library\share"

	labels, coords = zip(*objects)
	lats, longs = zip(*coords)
	points = pd.DataFrame({'Label': labels, 'Latitude': lats, 'Longitude': longs, 'Coordinates': coords})
	points['Coordinates'] = points['Coordinates'].apply(Point)
	points_df = gpd.GeoDataFrame(points, geometry='Coordinates')
	points_df.crs = {'init':'espg:4326'}
	# points_df.to_crs({'init':'espg:2249'})

	print(points_df.head())
	print(points_df.crs)

	buildings_df = gpd.read_file('BASEMAP_Buildings/BASEMAP_Buildings.shp')
	# buildings_df = gpd.read_file('structures_poly_49/structures_poly_49.shp')
	print(buildings_df.head())
	buildings_df = buildings_df.to_crs({'init':'espg:4326'})
	print(buildings_df.crs)
	ax = buildings_df.plot()

	points_df.plot(ax=ax, color='red')

	plt.show()

objects = [('DROID', (42.3588031, -71.0946304))]
map_pds(objects)