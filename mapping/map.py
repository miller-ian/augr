##given a list of (object, (lat,lon)), create a map of the objects
##includes a 'DROID' object, which is the location of the module itself
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def map(objects):
	objs = dict(objects)

	my_map = Basemap(projection='merc', resolution='i',
			urcrnrlon=-71.060568,
			urcrnrlat=42.376264,
			llcrnrlon=-71.119432,
			llcrnrlat=42.321144

		)

	# my_map.drawcoastlines()
	# my_map.drawrivers()
	# my_map.drawcounties()
	# my_map.bluemarble()

	# my_map.readshapefile('BASEMAP_Buildings/BASEMAP_Buildings','buildings')
	my_map.readshapefile('BASEMAP_Roads/BASEMAP_Roads','buildings')
	# my_map.readshapefile('structures_poly_49/structures_poly_49', 'buildings')

	my_map.plot(objs['DROID'][0],objs['DROID'][1])

 
	plt.show()

objects = [('DROID', (42.3588031, -71.0946304))]
map(objects)