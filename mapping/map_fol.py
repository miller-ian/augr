import folium
import os

##objects = list of item; [(label, (long, lat))]
url = 'render.html'

icon_map = {'DROID': ('tablet','blue'), 'person': ('male','red'), 'cat': ('paw','green'), 'dog': ('paw','green'), 'cow': ('paw','green'), 'car': ('car','orange')}

def map_fol(objects):
	m = folium.Map(location=[42.360, -71.087], zoom_start=18)
	for label, coords in objects:
		lon, lat = coords
		marker = folium.map.Icon(
			color=icon_map[label][1],
			icon=icon_map[label][0],
			prefix='fa'
			)
		folium.map.Marker(
			location=[lon,lat],
			tooltip=label,
			icon=marker).add_to(m)

	m.save(url)
	os.system("open "+ url)

objects = [('DROID', (42.36, -71.087))]
map_fol(objects)
