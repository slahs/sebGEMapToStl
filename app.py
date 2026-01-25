"""
Map-to-STL v2.2 - Bambu Lab Style
Created by SebGE - https://makerworld.com/de/@SebGE
Now with Markers (Heart, Star, etc.) + Gap for multi-color printing!
"""
import os,io,requests,math
from flask import Flask,request,jsonify,send_file,render_template_string
from flask_cors import CORS
import geopandas as gpd
import numpy as np
from shapely.geometry import box,MultiPolygon,Polygon,LineString,Point
from shapely.ops import unary_union
from shapely.affinity import translate,scale,rotate
import trimesh

app=Flask(__name__)
CORS(app)

# ==================== MARKER SHAPES ====================

def create_heart(size=10):
    """Create a heart shape polygon."""
    t = np.linspace(0, 2*np.pi, 100)
    x = 16 * np.sin(t)**3
    y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    # Normalize to size
    x = x / 32 * size
    y = y / 32 * size + size*0.1  # Shift up slightly
    coords = list(zip(x, y))
    return Polygon(coords)

def create_star(size=10, points=5):
    """Create a star shape polygon."""
    outer_r = size / 2
    inner_r = outer_r * 0.4
    angles = np.linspace(-np.pi/2, 3*np.pi/2, points*2+1)[:-1]
    coords = []
    for i, angle in enumerate(angles):
        r = outer_r if i % 2 == 0 else inner_r
        coords.append((r * np.cos(angle), r * np.sin(angle)))
    return Polygon(coords)

def create_cross(size=10):
    """Create a cross/plus shape polygon."""
    w = size * 0.3  # Width of arms
    h = size / 2    # Half length
    return Polygon([
        (-w/2, -h), (w/2, -h), (w/2, -w/2), (h, -w/2), (h, w/2), (w/2, w/2),
        (w/2, h), (-w/2, h), (-w/2, w/2), (-h, w/2), (-h, -w/2), (-w/2, -w/2), (-w/2, -h)
    ])

def create_pin(size=10):
    """Create a map pin/location marker shape."""
    # Teardrop shape
    t = np.linspace(0, 2*np.pi, 50)
    r = size/2 * (1 - 0.3*np.cos(t))
    x = r * np.sin(t)
    y = r * np.cos(t) * 0.7 + size*0.15
    coords = list(zip(x, y))
    return Polygon(coords)

def create_house(size=10):
    """Create a simple house shape."""
    h = size / 2
    w = size * 0.4
    roof_h = size * 0.35
    return Polygon([
        (-w, -h), (w, -h), (w, h*0.3), (0, h*0.3 + roof_h), (-w, h*0.3), (-w, -h)
    ])

def create_circle(size=10):
    """Create a circle shape."""
    t = np.linspace(0, 2*np.pi, 60)
    r = size / 2
    coords = [(r*np.cos(a), r*np.sin(a)) for a in t]
    return Polygon(coords)

def create_diamond(size=10):
    """Create a diamond shape."""
    h = size / 2
    return Polygon([(0, h), (h*0.6, 0), (0, -h), (-h*0.6, 0)])

MARKERS = {
    'heart': create_heart,
    'star': create_star,
    'cross': create_cross,
    'pin': create_pin,
    'house': create_house,
    'circle': create_circle,
    'diamond': create_diamond,
    'none': None
}

def smooth_line(coords,tol=0.5):
    if len(coords)<3:return coords
    line=LineString(coords)
    s=list(line.simplify(tol,preserve_topology=True).coords)
    if len(s)<3:return s
    for _ in range(2):
        n=[s[0]]
        for i in range(len(s)-1):
            p0,p1=s[i],s[i+1]
            n.extend([(0.75*p0[0]+0.25*p1[0],0.75*p0[1]+0.25*p1[1]),(0.25*p0[0]+0.75*p1[0],0.25*p0[1]+0.75*p1[1])])
        n.append(s[-1]);s=n
    return s

def geocode(q):
    known={'berlin':(52.52,13.405),'düsseldorf':(51.2277,6.7735),'königsallee':(51.2194,6.7784),'münchen':(48.1351,11.582),'hamburg':(53.5511,9.9937),'köln':(50.9375,6.9603),'frankfurt':(50.1109,8.6821),'paris':(48.8566,2.3522),'london':(51.5074,-0.1278),'new york':(40.7128,-74.006),'amsterdam':(52.3676,4.9041),'wien':(48.2082,16.3738),'zürich':(47.3769,8.5417),'tokyo':(35.6762,139.6503),'alps':(46.8182,10.1673),'matterhorn':(45.9763,7.6586),'zugspitze':(47.4211,10.9853)}
    ql=q.lower().strip()
    for n,c in known.items():
        if n in ql:return{'lat':c[0],'lon':c[1],'name':q}
    try:
        p=ql.replace(',',' ').split()
        if len(p)>=2:
            lat,lon=float(p[0]),float(p[1])
            if -90<=lat<=90 and -180<=lon<=180:return{'lat':lat,'lon':lon,'name':f'{lat},{lon}'}
    except:pass
    try:
        r=requests.get('https://nominatim.openstreetmap.org/search',params={'q':q,'format':'json','limit':1},headers={'User-Agent':'MapToSTL/2.2'},timeout=10)
        if r.status_code==200 and r.json():
            d=r.json()[0];return{'lat':float(d['lat']),'lon':float(d['lon']),'name':d.get('display_name',q)}
    except:pass
    return None

def fetch_streets(lat,lon,radius,types=None):
    if types is None:types=['primary','secondary','tertiary','residential']
    tm={'motorway':'motorway|motorway_link','trunk':'trunk|trunk_link','primary':'primary|primary_link','secondary':'secondary|secondary_link','tertiary':'tertiary|tertiary_link','residential':'residential|living_street','unclassified':'unclassified','service':'service','pedestrian':'pedestrian','footway':'footway|steps','cycleway':'cycleway','track':'track','bridleway':'bridleway','path':'path'}
    tags=[tm[t] for t in types if t in tm]
    if not tags:tags=['residential']
    regex='|'.join(tags);bbox=radius/111000
    q=f'[out:json][timeout:60];(way["highway"~"^({regex})$"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5}););out body;>;out skel qt;'
    try:
        r=requests.post('https://overpass-api.de/api/interpreter',data={'data':q},timeout=60)
        if r.status_code!=200:raise Exception(f"API {r.status_code}")
        data=r.json();nodes={e['id']:(e['lon'],e['lat']) for e in data.get('elements',[]) if e['type']=='node'}
        lines=[];tol=0.00001*(radius/500)
        for e in data.get('elements',[]):
            if e['type']=='way' and 'nodes' in e:
                coords=[nodes[n] for n in e['nodes'] if n in nodes]
                if len(coords)>=2:
                    sm=smooth_line(coords,tol)
                    if len(sm)>=2:lines.append(LineString(sm))
        if not lines:raise Exception("No streets")
        print(f"  ✓ {len(lines)} streets");return gpd.GeoDataFrame(geometry=lines,crs='EPSG:4326')
    except Exception as e:
        print(f"  ✗ {e}");import random;random.seed(int(abs(lat*10000+lon*1000))%(2**31))
        dl,dlo=radius/111320,radius/111320/np.cos(np.radians(lat));lines=[]
        for i in range(6):
            y=lat-dl+(2*dl*i/5);pts=[(lon-dlo+(2*dlo*j/20),y+random.uniform(-dl*0.02,dl*0.02)) for j in range(21)];lines.append(LineString(pts))
            x=lon-dlo+(2*dlo*i/5);pts=[(x+random.uniform(-dlo*0.02,dlo*0.02),lat-dl+(2*dl*j/20)) for j in range(21)];lines.append(LineString(pts))
        return gpd.GeoDataFrame(geometry=lines,crs='EPSG:4326')

def fetch_elevation(lat, lon, radius, resolution=100):
    deg_lat = radius / 111320
    deg_lon = deg_lat / np.cos(np.radians(lat))
    lats = np.linspace(lat - deg_lat, lat + deg_lat, resolution)
    lons = np.linspace(lon - deg_lon, lon + deg_lon, resolution)
    try:
        locations = []
        sample_res = min(10, resolution)
        sample_lats = np.linspace(lat - deg_lat, lat + deg_lat, sample_res)
        sample_lons = np.linspace(lon - deg_lon, lon + deg_lon, sample_res)
        for la in sample_lats:
            for lo in sample_lons:
                locations.append({"latitude": la, "longitude": lo})
        response = requests.post("https://api.open-elevation.com/api/v1/lookup",json={"locations": locations},timeout=30)
        if response.status_code == 200:
            data = response.json()
            elevations = [r["elevation"] for r in data["results"]]
            elev_grid = np.array(elevations).reshape(sample_res, sample_res)
            from scipy.ndimage import zoom
            zoom_factor = resolution / sample_res
            elevation_data = zoom(elev_grid, zoom_factor, order=3)
            print(f"  ✓ Elevation data fetched")
            return elevation_data, lats, lons
    except Exception as e:
        print(f"  ✗ Elevation API error: {e}")
    print("  → Generating procedural terrain")
    return generate_procedural_terrain(lat, lon, resolution), lats, lons

def generate_procedural_terrain(lat, lon, resolution):
    np.random.seed(int(abs(lat * 1000 + lon * 100)) % (2**31))
    x = np.linspace(0, 4, resolution)
    y = np.linspace(0, 4, resolution)
    X, Y = np.meshgrid(x, y)
    terrain = np.zeros((resolution, resolution))
    for freq, amp in [(1, 500), (2, 250), (4, 125), (8, 62)]:
        phase_x = np.random.rand() * 10
        phase_y = np.random.rand() * 10
        terrain += amp * (np.sin(freq * X + phase_x) * np.cos(freq * Y + phase_y) + 1) / 2
    for _ in range(3):
        px, py = np.random.rand() * resolution, np.random.rand() * resolution
        dist = np.sqrt((np.arange(resolution)[:, None] - px)**2 + (np.arange(resolution)[None, :] - py)**2)
        terrain += 300 * np.exp(-dist**2 / (resolution/3)**2)
    terrain = 200 + (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 1800
    return terrain

def create_terrain_mesh(elevation_data, size, height_scale, base_height=2.0):
    resolution = elevation_data.shape[0]
    elev_min = elevation_data.min()
    elev_max = elevation_data.max()
    elev_range = elev_max - elev_min if elev_max > elev_min else 1
    normalized = (elevation_data - elev_min) / elev_range * height_scale
    vertices = []
    faces = []
    step = size / (resolution - 1)
    half = size / 2
    for i in range(resolution):
        for j in range(resolution):
            x = -half + j * step
            y = -half + i * step
            z = base_height + normalized[i, j]
            vertices.append([x, y, z])
    for i in range(resolution):
        for j in range(resolution):
            x = -half + j * step
            y = -half + i * step
            vertices.append([x, y, 0])
    n = resolution
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * n + j
            v1 = i * n + j + 1
            v2 = (i + 1) * n + j
            v3 = (i + 1) * n + j + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    offset = resolution * resolution
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = offset + i * n + j
            v1 = offset + i * n + j + 1
            v2 = offset + (i + 1) * n + j
            v3 = offset + (i + 1) * n + j + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    for j in range(resolution - 1):
        top1, top2 = j, j + 1
        bot1, bot2 = offset + j, offset + j + 1
        faces.append([top1, top2, bot1])
        faces.append([top2, bot2, bot1])
    for j in range(resolution - 1):
        top1 = (resolution - 1) * n + j
        top2 = (resolution - 1) * n + j + 1
        bot1 = offset + (resolution - 1) * n + j
        bot2 = offset + (resolution - 1) * n + j + 1
        faces.append([top1, bot1, top2])
        faces.append([top2, bot1, bot2])
    for i in range(resolution - 1):
        top1 = i * n
        top2 = (i + 1) * n
        bot1 = offset + i * n
        bot2 = offset + (i + 1) * n
        faces.append([top1, bot1, top2])
        faces.append([top2, bot1, bot2])
    for i in range(resolution - 1):
        top1 = i * n + (resolution - 1)
        top2 = (i + 1) * n + (resolution - 1)
        bot1 = offset + i * n + (resolution - 1)
        bot2 = offset + (i + 1) * n + (resolution - 1)
        faces.append([top1, top2, bot1])
        faces.append([top2, bot2, bot1])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh

def gen_svg(gdf,lat,lon,radius,lw,size,style='lines',bg='#1a1a1a',fg='#00AE42',marker_type='none',marker_size=10,marker_gap=2):
    gp=gdf.to_crs(gdf.estimate_utm_crs())
    ctr=gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon],[lat]),crs='EPSG:4326').to_crs(gp.crs).geometry.iloc[0]
    bb=box(ctr.x-radius,ctr.y-radius,ctr.x+radius,ctr.y+radius);cl=gp.clip(bb)
    if cl.empty:return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}"><rect width="100%" height="100%" fill="{bg}"/></svg>'
    sc=size/(2*radius)
    def tx(x,y):return(x-ctr.x)*sc+size/2,size/2-(y-ctr.y)*sc
    
    # Create marker if specified
    marker_svg = ""
    gap_poly = None
    if marker_type != 'none' and marker_type in MARKERS and MARKERS[marker_type]:
        marker = MARKERS[marker_type](marker_size)
        marker = translate(marker, size/2, size/2)  # Center it
        
        # Create gap polygon (marker + buffer)
        if marker_gap > 0:
            gap_poly = marker.buffer(marker_gap)
        
        # SVG path for marker
        coords = list(marker.exterior.coords)
        marker_svg = f'<path d="M{coords[0][0]:.1f},{coords[0][1]:.1f}'
        for c in coords[1:]:
            marker_svg += f' L{c[0]:.1f},{c[1]:.1f}'
        marker_svg += f'Z" fill="{fg}" stroke="{fg}" stroke-width="0.5"/>'
    
    paths=[]
    for g in cl.geometry:
        if g.is_empty:continue
        for ln in(g.geoms if g.geom_type=='MultiLineString' else[g]):
            if ln.geom_type!='LineString':continue
            c=list(ln.coords)
            if len(c)<2:continue
            
            # Transform coordinates
            transformed = [(tx(p[0], p[1])) for p in c]
            
            # If gap_poly exists, clip the line
            if gap_poly:
                line_svg = LineString(transformed)
                diff = line_svg.difference(gap_poly)
                if diff.is_empty:
                    continue
                # Handle multi-part result
                if diff.geom_type == 'MultiLineString':
                    for part in diff.geoms:
                        if len(part.coords) >= 2:
                            pc = list(part.coords)
                            d = f"M{pc[0][0]:.1f},{pc[0][1]:.1f}"
                            for p in pc[1:]:
                                d += f" L{p[0]:.1f},{p[1]:.1f}"
                            paths.append(d)
                elif diff.geom_type == 'LineString' and len(diff.coords) >= 2:
                    pc = list(diff.coords)
                    d = f"M{pc[0][0]:.1f},{pc[0][1]:.1f}"
                    for p in pc[1:]:
                        d += f" L{p[0]:.1f},{p[1]:.1f}"
                    paths.append(d)
            else:
                d=f"M{transformed[0][0]:.1f},{transformed[0][1]:.1f}"
                for p in transformed[1:]:d+=f" L{p[0]:.1f},{p[1]:.1f}"
                paths.append(d)
    
    if style=='negative':
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}mm" height="{size}mm"><defs><mask id="m"><rect width="100%" height="100%" fill="white"/><g stroke="black" stroke-width="{lw}" stroke-linecap="round" fill="none">'
        for p in paths:svg+=f'<path d="{p}"/>'
        svg+=f'</g></mask></defs><rect width="100%" height="100%" fill="{fg}" mask="url(#m)"/>{marker_svg}</svg>'
    elif style=='filled':
        buf=lw/sc/2;uni=unary_union(cl.geometry.buffer(buf,cap_style=2,join_style=2))
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}mm" height="{size}mm"><rect width="100%" height="100%" fill="{bg}"/><g fill="{fg}">'
        for poly in(uni.geoms if hasattr(uni,'geoms')else[uni]):
            if poly.is_empty:continue
            d="M"+" L".join(f"{tx(*c)[0]:.1f},{tx(*c)[1]:.1f}" for c in poly.exterior.coords)+"Z"
            for h in poly.interiors:d+=" M"+" L".join(f"{tx(*c)[0]:.1f},{tx(*c)[1]:.1f}" for c in h.coords)+"Z"
            svg+=f'<path d="{d}" fill-rule="evenodd"/>'
        svg+=f'</g>{marker_svg}</svg>'
    elif style=='outline':
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}mm" height="{size}mm"><defs><clipPath id="c"><circle cx="{size/2}" cy="{size/2}" r="{size/2-2}"/></clipPath></defs><rect width="100%" height="100%" fill="{bg}"/><g clip-path="url(#c)"><g stroke="{fg}" stroke-width="{lw}" stroke-linecap="round" fill="none">'
        for p in paths:svg+=f'<path d="{p}"/>'
        svg+=f'</g></g><circle cx="{size/2}" cy="{size/2}" r="{size/2-1}" fill="none" stroke="{fg}" stroke-width="1.5"/>{marker_svg}</svg>'
    elif style=='transparent':
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}mm" height="{size}mm"><g stroke="{fg}" stroke-width="{lw}" stroke-linecap="round" fill="none">'
        for p in paths:svg+=f'<path d="{p}"/>'
        svg+=f'</g>{marker_svg}</svg>'
    else:
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}mm" height="{size}mm"><rect width="100%" height="100%" fill="{bg}"/><g stroke="{fg}" stroke-width="{lw}" stroke-linecap="round" fill="none">'
        for p in paths:svg+=f'<path d="{p}"/>'
        svg+=f'</g>{marker_svg}</svg>'
    return svg

def gen_stl(gdf,lat,lon,radius,lw,size,height,neg=False,marker_type='none',marker_size=10,marker_gap=2,marker_height=None):
    if marker_height is None:
        marker_height = height
    
    gp=gdf.to_crs(gdf.estimate_utm_crs())
    ctr=gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon],[lat]),crs='EPSG:4326').to_crs(gp.crs).geometry.iloc[0]
    bb=box(ctr.x-radius,ctr.y-radius,ctr.x+radius,ctr.y+radius);cl=gp.clip(bb)
    if cl.empty:raise ValueError("No streets")
    sc=size/(2*radius);buf=lw/sc/2
    
    # Create marker geometry
    marker_poly = None
    gap_poly = None
    if marker_type != 'none' and marker_type in MARKERS and MARKERS[marker_type]:
        marker_poly = MARKERS[marker_type](marker_size)  # Already centered at 0,0
        if marker_gap > 0:
            gap_poly = marker_poly.buffer(marker_gap)
    
    uni=unary_union(cl.geometry.buffer(buf,cap_style=2,join_style=2)).simplify(0.5,preserve_topology=True)
    
    def norm(poly):
        c=np.array(poly.exterior.coords);c[:,0]=(c[:,0]-ctr.x)*sc;c[:,1]=(c[:,1]-ctr.y)*sc
        holes=[]
        for i in poly.interiors:hc=np.array(i.coords);hc[:,0]=(hc[:,0]-ctr.x)*sc;hc[:,1]=(hc[:,1]-ctr.y)*sc;holes.append(hc.tolist())
        return Polygon(c.tolist(),holes)
    
    polys=[norm(p) for p in(uni.geoms if hasattr(uni,'geoms')else[uni])]
    streets=MultiPolygon(polys) if len(polys)>1 else polys[0]
    
    # Cut gap from streets if marker with gap
    if gap_poly:
        streets = streets.difference(gap_poly)
    
    if neg:
        result=box(-size/2,-size/2,size/2,size/2).difference(streets)
    else:
        result = streets
    
    meshes=[]
    
    # Add street mesh
    for poly in(result.geoms if hasattr(result,'geoms')else[result]):
        if poly.is_empty or poly.area<0.1:continue
        try:
            m=trimesh.creation.extrude_polygon(poly,height)
            if m and len(m.vertices)>0:meshes.append(m)
        except:continue
    
    # Add marker mesh (separate, so can be different color in slicer)
    if marker_poly:
        try:
            marker_mesh = trimesh.creation.extrude_polygon(marker_poly, marker_height)
            if marker_mesh and len(marker_mesh.vertices) > 0:
                meshes.append(marker_mesh)
        except Exception as e:
            print(f"  Marker mesh error: {e}")
    
    if not meshes:raise ValueError("No mesh")
    c=trimesh.util.concatenate(meshes);c.fill_holes();c.fix_normals();return c

def gen_stl_separate(gdf,lat,lon,radius,lw,size,height,marker_type='none',marker_size=10,marker_gap=2,marker_height=None):
    """Generate separate STL files for streets and marker (for multi-color printing)."""
    if marker_height is None:
        marker_height = height
    
    gp=gdf.to_crs(gdf.estimate_utm_crs())
    ctr=gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon],[lat]),crs='EPSG:4326').to_crs(gp.crs).geometry.iloc[0]
    bb=box(ctr.x-radius,ctr.y-radius,ctr.x+radius,ctr.y+radius);cl=gp.clip(bb)
    if cl.empty:raise ValueError("No streets")
    sc=size/(2*radius);buf=lw/sc/2
    
    marker_poly = None
    gap_poly = None
    if marker_type != 'none' and marker_type in MARKERS and MARKERS[marker_type]:
        marker_poly = MARKERS[marker_type](marker_size)
        if marker_gap > 0:
            gap_poly = marker_poly.buffer(marker_gap)
    
    uni=unary_union(cl.geometry.buffer(buf,cap_style=2,join_style=2)).simplify(0.5,preserve_topology=True)
    
    def norm(poly):
        c=np.array(poly.exterior.coords);c[:,0]=(c[:,0]-ctr.x)*sc;c[:,1]=(c[:,1]-ctr.y)*sc
        holes=[]
        for i in poly.interiors:hc=np.array(i.coords);hc[:,0]=(hc[:,0]-ctr.x)*sc;hc[:,1]=(hc[:,1]-ctr.y)*sc;holes.append(hc.tolist())
        return Polygon(c.tolist(),holes)
    
    polys=[norm(p) for p in(uni.geoms if hasattr(uni,'geoms')else[uni])]
    streets=MultiPolygon(polys) if len(polys)>1 else polys[0]
    
    if gap_poly:
        streets = streets.difference(gap_poly)
    
    # Streets mesh
    street_meshes=[]
    for poly in(streets.geoms if hasattr(streets,'geoms')else[streets]):
        if poly.is_empty or poly.area<0.1:continue
        try:
            m=trimesh.creation.extrude_polygon(poly,height)
            if m and len(m.vertices)>0:street_meshes.append(m)
        except:continue
    
    street_mesh = trimesh.util.concatenate(street_meshes) if street_meshes else None
    if street_mesh:
        street_mesh.fill_holes()
        street_mesh.fix_normals()
    
    # Marker mesh
    marker_mesh = None
    if marker_poly:
        try:
            marker_mesh = trimesh.creation.extrude_polygon(marker_poly, marker_height)
            if marker_mesh:
                marker_mesh.fill_holes()
                marker_mesh.fix_normals()
        except:pass
    
    return street_mesh, marker_mesh

HTML='''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Map to STL Generator | by SebGE</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/><style>:root{--b:#00AE42;--bd:#009639;--bl:#00C853;--bg:#0d0d0d;--c:#1a1a1a;--i:#252525;--br:#333;--t:#fafafa;--d:#888}*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--t);min-height:100vh}.app{display:grid;grid-template-columns:440px 1fr 320px;min-height:100vh}.sb{background:var(--c);border-right:1px solid var(--br);padding:16px;display:flex;flex-direction:column;gap:12px;overflow-y:auto}.hd{display:flex;align-items:center;gap:12px;padding-bottom:12px;border-bottom:1px solid var(--br)}.logo{width:44px;height:44px;background:linear-gradient(135deg,var(--b),var(--bd));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:22px}.hd h1{font-size:18px;font-weight:700}.hd .sub{font-size:10px;color:var(--b);font-weight:500}.cr{display:flex;align-items:center;gap:12px;background:linear-gradient(135deg,rgba(0,174,66,.15),rgba(0,174,66,.05));border:1px solid var(--b);border-radius:10px;padding:12px 14px;text-decoration:none;transition:all .2s;position:relative;overflow:hidden}.cr::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent,rgba(0,174,66,.1),transparent);transform:translateX(-100%);transition:transform .5s}.cr:hover::before{transform:translateX(100%)}.cr:hover{background:rgba(0,174,66,.2);transform:translateY(-2px);box-shadow:0 4px 15px rgba(0,174,66,.3)}.cr-av{width:40px;height:40px;background:linear-gradient(135deg,var(--b),var(--bd));border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;color:#fff;box-shadow:0 2px 8px rgba(0,174,66,.4)}.cr-i{flex:1}.cr-n{font-size:13px;font-weight:700;color:var(--t)}.cr-l{font-size:11px;color:var(--b);margin-top:2px}.cr-bd{display:flex;flex-direction:column;align-items:flex-end;gap:2px}.cr-tag{font-size:8px;background:var(--b);color:#fff;padding:3px 8px;border-radius:4px;font-weight:700;letter-spacing:.5px}.cr-mw{font-size:9px;color:var(--d)}.sw{position:relative}.si{width:100%;padding:12px 14px 12px 40px;background:var(--i);border:1px solid var(--br);border-radius:8px;color:var(--t);font-size:13px}.si:focus{outline:none;border-color:var(--b)}.sic{position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--d)}.sbt{position:absolute;right:6px;top:50%;transform:translateY(-50%);padding:6px 14px;background:var(--b);border:none;border-radius:6px;color:#fff;font-size:11px;font-weight:600;cursor:pointer}.sbt:hover{background:var(--bd)}.cds{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.cd{background:var(--i);border:1px solid var(--br);border-radius:6px;padding:8px 10px;font-family:monospace;font-size:11px}.cd label{display:block;font-size:8px;color:var(--d);text-transform:uppercase;margin-bottom:2px;font-family:sans-serif}.cd span{color:var(--b);font-weight:500}.sec{background:var(--i);border:1px solid var(--br);border-radius:10px;padding:14px}.st{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--b);margin-bottom:12px;display:flex;align-items:center;gap:8px}.st::before{content:"";width:4px;height:14px;background:var(--b);border-radius:2px}.mode-tabs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}.mode-tab{padding:12px;background:var(--c);border:2px solid var(--br);border-radius:8px;cursor:pointer;text-align:center;transition:all .15s}.mode-tab:hover{border-color:var(--b)}.mode-tab.a{border-color:var(--b);background:rgba(0,174,66,.15)}.mode-tab .ic{font-size:24px;margin-bottom:4px}.mode-tab .nm{font-size:11px;font-weight:600}.mode-tab .ds{font-size:9px;color:var(--d);margin-top:2px}.prs{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}.pr{padding:6px 10px;background:var(--c);border:1px solid var(--br);border-radius:6px;color:var(--t);font-size:10px;cursor:pointer;font-weight:500}.pr:hover{border-color:var(--b);color:var(--b)}.pr.a{background:var(--b);border-color:var(--b);color:#fff}.tps{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.tp{display:flex;align-items:center;gap:5px;padding:6px 8px;background:var(--c);border:1px solid var(--br);border-radius:6px;cursor:pointer;font-size:10px}.tp:hover{border-color:var(--b)}.tp.a{border-color:var(--b);background:rgba(0,174,66,.1)}.tp .ck{width:12px;height:12px;border:2px solid var(--br);border-radius:3px;font-size:8px;display:flex;align-items:center;justify-content:center}.tp.a .ck{background:var(--b);border-color:var(--b);color:#fff}.stl{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}.sty{padding:10px 6px;background:var(--c);border:2px solid var(--br);border-radius:8px;cursor:pointer;text-align:center}.sty:hover{border-color:var(--b)}.sty.a{border-color:var(--b);background:rgba(0,174,66,.1)}.sty .ic{font-size:20px}.sty .nm{font-size:9px;margin-top:4px;font-weight:500}.markers{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}.marker{padding:10px 6px;background:var(--c);border:2px solid var(--br);border-radius:8px;cursor:pointer;text-align:center;transition:all .15s}.marker:hover{border-color:var(--b)}.marker.a{border-color:var(--b);background:rgba(0,174,66,.1)}.marker .ic{font-size:22px}.marker .nm{font-size:8px;margin-top:4px;font-weight:500}.cls{display:flex;gap:10px;margin-top:10px}.clp{display:flex;align-items:center;gap:6px;flex:1}.clp label{font-size:10px;color:var(--d);font-weight:500}.clp input[type="color"]{width:32px;height:24px;border:none;border-radius:4px;cursor:pointer;background:none}.cld{width:18px;height:18px;border-radius:4px;cursor:pointer;border:2px solid transparent}.cld:hover{transform:scale(1.15)}.pm{margin-bottom:10px}.pmh{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}.pml{font-size:11px;font-weight:500}.pmv{font-family:monospace;font-size:10px;color:var(--b);background:var(--c);padding:3px 8px;border-radius:4px;font-weight:600}input[type="range"]{width:100%;height:6px;background:var(--c);border-radius:3px;-webkit-appearance:none;margin-top:2px}input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;background:var(--b);border-radius:50%;cursor:pointer}.ex{margin-top:auto;padding-top:14px;border-top:1px solid var(--br)}.bts{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px}.btn{padding:12px;border:none;border-radius:8px;font-size:11px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px}.bp{background:linear-gradient(135deg,var(--b),var(--bd));color:#fff;box-shadow:0 4px 12px rgba(0,174,66,.3)}.bp:hover{transform:translateY(-2px)}.bs{background:var(--i);color:var(--t);border:1px solid var(--br)}.bs:hover{border-color:var(--b);color:var(--b)}.btn:disabled{opacity:.5;cursor:not-allowed}.sp{width:14px;height:14px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .7s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.sts{padding:10px;border-radius:6px;font-size:11px;margin-top:8px;display:none;font-weight:500}.sts.er{display:block;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#ef4444}.sts.ok{display:block;background:rgba(0,174,66,.1);border:1px solid rgba(0,174,66,.3);color:var(--b)}.mw{position:relative}#map{width:100%;height:100%;background:var(--i)}.mi{position:absolute;top:12px;right:12px;background:rgba(26,26,26,.95);border:1px solid var(--br);border-radius:8px;padding:10px 14px;font-size:10px;color:var(--d);z-index:1000}.pv{background:var(--c);border-left:1px solid var(--br);padding:16px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}.pv h2{font-size:13px;font-weight:700}.bdg{font-size:9px;padding:3px 8px;background:var(--b);color:#fff;border-radius:6px;margin-left:auto;font-weight:600}.pvb{background:var(--i);border:1px solid var(--br);border-radius:12px;aspect-ratio:1;display:flex;align-items:center;justify-content:center;overflow:hidden}#pvs{width:100%;height:100%}.ph{color:var(--d);font-size:11px;text-align:center}.sts2{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.sta{background:var(--i);border:1px solid var(--br);border-radius:8px;padding:8px;text-align:center}.sta label{font-size:8px;color:var(--d);text-transform:uppercase}.sta .vl{font-family:monospace;font-size:12px;color:var(--b);margin-top:3px;font-weight:600}.inf{font-size:10px;color:var(--d);line-height:1.5;background:var(--i);border-radius:8px;padding:12px}.inf strong{color:var(--b)}.ft{margin-top:auto;padding-top:12px;border-top:1px solid var(--br);text-align:center;font-size:9px;color:var(--d)}.ft a{color:var(--b);text-decoration:none;font-weight:600}.ft a:hover{text-decoration:underline}.terrain-opts,.marker-opts{display:none}.terrain-opts.show,.marker-opts.show{display:block}.chk{display:flex;align-items:center;gap:8px;padding:8px 0;cursor:pointer}.chk input{display:none}.chk .box{width:18px;height:18px;border:2px solid var(--br);border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:12px;transition:all .15s}.chk input:checked+.box{background:var(--b);border-color:var(--b);color:#fff}.chk span{font-size:11px}@media(max-width:1300px){.app{grid-template-columns:420px 1fr}.pv{display:none}}@media(max-width:700px){.app{grid-template-columns:1fr;grid-template-rows:35vh auto}.sb{order:2}.mw{order:1}}</style></head><body><div class="app"><aside class="sb"><div class="hd"><div class="logo">🗺️</div><div><h1>Map → STL</h1><div class="sub">3D PRINT YOUR WORLD</div></div></div><a href="https://makerworld.com/de/@SebGE" target="_blank" class="cr"><div class="cr-av">S</div><div class="cr-i"><div class="cr-n">Created by SebGE</div><div class="cr-l">Check out my other models!</div></div><div class="cr-bd"><div class="cr-tag">✓ MAKER</div><div class="cr-mw">MakerWorld</div></div></a><div><div class="sw"><svg class="sic" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><input type="text" id="loc" class="si" placeholder="Search city or mountain..." value="Düsseldorf"><button class="sbt" onclick="search()">Search</button></div><div class="cds"><div class="cd"><label>Lat</label><span id="lv">51.2277</span></div><div class="cd"><label>Lon</label><span id="lov">6.7735</span></div></div></div><div class="sec"><div class="st">Map Mode</div><div class="mode-tabs"><div class="mode-tab a" data-mode="streets" onclick="setMode('streets')"><div class="ic">🛣️</div><div class="nm">Street Map</div><div class="ds">Roads & paths</div></div><div class="mode-tab" data-mode="terrain" onclick="setMode('terrain')"><div class="ic">🏔️</div><div class="nm">3D Terrain</div><div class="ds">Mountains & hills</div></div></div></div><div class="sec street-opts"><div class="st">Center Marker</div><div class="markers"><div class="marker a" data-m="none" onclick="setMarker('none')"><div class="ic">❌</div><div class="nm">None</div></div><div class="marker" data-m="heart" onclick="setMarker('heart')"><div class="ic">❤️</div><div class="nm">Heart</div></div><div class="marker" data-m="star" onclick="setMarker('star')"><div class="ic">⭐</div><div class="nm">Star</div></div><div class="marker" data-m="pin" onclick="setMarker('pin')"><div class="ic">📍</div><div class="nm">Pin</div></div><div class="marker" data-m="house" onclick="setMarker('house')"><div class="ic">🏠</div><div class="nm">House</div></div><div class="marker" data-m="cross" onclick="setMarker('cross')"><div class="ic">✚</div><div class="nm">Cross</div></div><div class="marker" data-m="circle" onclick="setMarker('circle')"><div class="ic">⭕</div><div class="nm">Circle</div></div><div class="marker" data-m="diamond" onclick="setMarker('diamond')"><div class="ic">💎</div><div class="nm">Diamond</div></div></div><div class="marker-opts show"><div class="pm" style="margin-top:10px"><div class="pmh"><span class="pml">Marker Size</span><span class="pmv" id="msv">12mm</span></div><input type="range" id="ms" min="5" max="30" value="12" step="1" oninput="$('msv').textContent=this.value+'mm'"></div><div class="pm"><div class="pmh"><span class="pml">Gap (for multi-color)</span><span class="pmv" id="mgv">2mm</span></div><input type="range" id="mg" min="0" max="5" value="2" step="0.5" oninput="$('mgv').textContent=this.value+'mm'"></div><label class="chk"><input type="checkbox" id="sep" checked><span class="box">✓</span><span>Export marker separately (for Bambu multi-color)</span></label></div></div><div class="sec street-opts"><div class="st">Street Types</div><div class="prs"><button class="pr a" onclick="preset('city')">🏙️ City</button><button class="pr" onclick="preset('rural')">🌾 Rural</button><button class="pr" onclick="preset('all')">📍 All</button><button class="pr" onclick="preset('main')">🛣️ Main</button><button class="pr" onclick="preset('paths')">🚶 Paths</button><button class="pr" onclick="preset('none')">❌ Clear</button></div><div class="tps" id="tps"><label class="tp a" data-t="motorway"><span class="ck">✓</span>🚀 Hwy</label><label class="tp a" data-t="trunk"><span class="ck">✓</span>🛣️ Trunk</label><label class="tp a" data-t="primary"><span class="ck">✓</span>🔴 Primary</label><label class="tp a" data-t="secondary"><span class="ck">✓</span>🟠 Second.</label><label class="tp a" data-t="tertiary"><span class="ck">✓</span>🟡 Tertiary</label><label class="tp a" data-t="residential"><span class="ck">✓</span>🏠 Resid.</label><label class="tp" data-t="unclassified"><span class="ck">✓</span>📍 Unclass.</label><label class="tp" data-t="service"><span class="ck">✓</span>🅿️ Service</label><label class="tp" data-t="track"><span class="ck">✓</span>🌾 Track</label><label class="tp" data-t="footway"><span class="ck">✓</span>🚶 Foot</label><label class="tp" data-t="cycleway"><span class="ck">✓</span>🚴 Cycle</label><label class="tp" data-t="path"><span class="ck">✓</span>🥾 Path</label></div></div><div class="sec street-opts"><div class="st">Output Style</div><div class="stl"><div class="sty a" data-s="lines" onclick="setStyle('lines')"><div class="ic">〰️</div><div class="nm">Lines</div></div><div class="sty" data-s="filled" onclick="setStyle('filled')"><div class="ic">⬛</div><div class="nm">Filled</div></div><div class="sty" data-s="negative" onclick="setStyle('negative')"><div class="ic">🔲</div><div class="nm">Negative</div></div><div class="sty" data-s="outline" onclick="setStyle('outline')"><div class="ic">⭕</div><div class="nm">Circle</div></div><div class="sty" data-s="transparent" onclick="setStyle('transparent')"><div class="ic">🔳</div><div class="nm">Trans</div></div></div><div class="cls"><div class="clp"><label>Lines:</label><input type="color" id="fg" value="#00AE42"><div class="cld" style="background:#00AE42" onclick="$('fg').value='#00AE42'"></div><div class="cld" style="background:#fff" onclick="$('fg').value='#ffffff'"></div></div><div class="clp"><label>BG:</label><input type="color" id="bg" value="#1a1a1a"><div class="cld" style="background:#1a1a1a" onclick="$('bg').value='#1a1a1a'"></div><div class="cld" style="background:#fff" onclick="$('bg').value='#ffffff'"></div></div></div></div><div class="sec terrain-opts"><div class="st">Terrain Settings</div><div class="pm"><div class="pmh"><span class="pml">Height Scale</span><span class="pmv" id="hsv">15mm</span></div><input type="range" id="hs" min="5" max="50" value="15" step="1" oninput="$('hsv').textContent=this.value+'mm'"></div><div class="pm"><div class="pmh"><span class="pml">Resolution</span><span class="pmv" id="resv">100</span></div><input type="range" id="res" min="50" max="200" value="100" step="10" oninput="$('resv').textContent=this.value"></div></div><div class="sec"><div class="st">Parameters</div><div class="pm"><div class="pmh"><span class="pml">Radius</span><span class="pmv" id="rv">500m</span></div><input type="range" id="rad" min="100" max="10000" value="500" step="100" oninput="$('rv').textContent=this.value>=1000?(this.value/1000)+'km':this.value+'m';uc()"></div><div class="pm street-opts"><div class="pmh"><span class="pml">Line Width</span><span class="pmv" id="lwv">1.2mm</span></div><input type="range" id="lw" min="0.3" max="4" value="1.2" step="0.1" oninput="$('lwv').textContent=parseFloat(this.value).toFixed(1)+'mm'"></div><div class="pm street-opts"><div class="pmh"><span class="pml">Height</span><span class="pmv" id="hv">2.0mm</span></div><input type="range" id="ht" min="0.4" max="8" value="2.0" step="0.1" oninput="$('hv').textContent=parseFloat(this.value).toFixed(1)+'mm'"></div><div class="pm"><div class="pmh"><span class="pml">Size</span><span class="pmv" id="sv">80mm</span></div><input type="range" id="sz" min="20" max="300" value="80" step="5" oninput="$('sv').textContent=this.value+'mm';$('ssz').textContent=this.value+'mm'"></div></div><div class="ex"><div class="bts"><button class="btn bs" id="bp" onclick="preview()">👁 Preview</button><button class="btn bs street-opts" id="bsv" onclick="expSVG()">📄 SVG</button><button class="btn bp" id="bst" onclick="expSTL()">⬇ STL</button></div><div class="bts street-opts"><button class="btn bs" style="font-size:10px" onclick="expSTL(false)">STL Normal</button><button class="btn bs" style="font-size:10px" onclick="expSTL(true)">STL Negative</button></div><div class="sts" id="sts"></div></div><div class="ft">Made with ❤️ by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a><br>Follow for more 3D printing tools!</div></aside><main class="mw"><div id="map"></div><div class="mi">🖱️ Click to set center</div></main><aside class="pv"><h2>Preview <span class="bdg" id="bdg" style="display:none">Ready</span></h2><div class="pvb"><div id="pvs"></div><div class="ph" id="ph">👁 Click Preview</div></div><div class="sts2"><div class="sta"><label>Data</label><div class="vl" id="sseg">—</div></div><div class="sta"><label>Size</label><div class="vl" id="ssz">80mm</div></div><div class="sta"><label>Marker</label><div class="vl" id="smk">None</div></div></div><div class="inf"><strong>🎨 Multi-Color Printing:</strong><br>• Enable "Export separately" for marker<br>• Load both STLs in Bambu Studio<br>• Assign different colors to each<br>• Gap ensures clean color separation!</div></aside></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>const $=id=>document.getElementById(id);let map,mk,cc,lat=51.2277,lon=6.7735,sty='lines',mode='streets',marker='none';const prs={city:['motorway','trunk','primary','secondary','tertiary','residential'],rural:['primary','secondary','tertiary','residential','track','unclassified','path','service'],all:['motorway','trunk','primary','secondary','tertiary','residential','unclassified','service','track','footway','cycleway','path'],main:['motorway','trunk','primary','secondary'],paths:['footway','cycleway','path','track'],none:[]};function init(){map=L.map('map').setView([lat,lon],14);L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{maxZoom:19}).addTo(map);um();map.on('click',e=>{lat=e.latlng.lat;lon=e.latlng.lng;um();uco()});document.querySelectorAll('.tp').forEach(el=>el.onclick=()=>{el.classList.toggle('a');document.querySelectorAll('.pr').forEach(p=>p.classList.remove('a'))});$('loc').onkeypress=e=>{if(e.key==='Enter')search()}}function setMode(m){mode=m;document.querySelectorAll('.mode-tab').forEach(el=>el.classList.toggle('a',el.dataset.mode===m));document.querySelectorAll('.street-opts').forEach(el=>el.style.display=m==='streets'?'':'none');document.querySelectorAll('.terrain-opts').forEach(el=>el.classList.toggle('show',m==='terrain'));if(m==='terrain'){$('rad').value=5000;$('rv').textContent='5km';uc()}}function setMarker(m){marker=m;document.querySelectorAll('.marker').forEach(el=>el.classList.toggle('a',el.dataset.m===m));document.querySelectorAll('.marker-opts').forEach(el=>el.classList.toggle('show',m!=='none'));$('smk').textContent=m==='none'?'None':m.charAt(0).toUpperCase()+m.slice(1)}function preset(n){const t=prs[n]||[];document.querySelectorAll('.tp').forEach(el=>el.classList.toggle('a',t.includes(el.dataset.t)));document.querySelectorAll('.pr').forEach(p=>p.classList.toggle('a',p.textContent.toLowerCase().includes(n)))}function setStyle(s){sty=s;document.querySelectorAll('.sty').forEach(el=>el.classList.toggle('a',el.dataset.s===s))}function gt(){return Array.from(document.querySelectorAll('.tp.a')).map(e=>e.dataset.t)}function um(){const r=+$('rad').value;if(mk)map.removeLayer(mk);if(cc)map.removeLayer(cc);mk=L.marker([lat,lon],{icon:L.divIcon({html:'<div style="width:16px;height:16px;background:#00AE42;border:2px solid #fff;border-radius:50%;box-shadow:0 2px 8px rgba(0,174,66,.5)"></div>',iconSize:[16,16],iconAnchor:[8,8]})}).addTo(map);cc=L.circle([lat,lon],{radius:r,color:'#00AE42',fillOpacity:.08,weight:2,dashArray:'8,8'}).addTo(map)}function uc(){const r=+$('rad').value;if(cc)cc.setRadius(r)}function uco(){$('lv').textContent=lat.toFixed(5);$('lov').textContent=lon.toFixed(5)}function pm(){return{lat,lon,radius:+$('rad').value,line_width:+$('lw').value,extrusion_height:+$('ht').value,target_size:+$('sz').value,street_types:gt(),style:sty,bg_color:$('bg').value,line_color:$('fg').value,mode,height_scale:+$('hs').value,resolution:+$('res').value,marker_type:marker,marker_size:+$('ms').value,marker_gap:+$('mg').value,separate_marker:$('sep').checked}}async function search(){const q=$('loc').value.trim();if(!q)return;try{const r=await fetch('/api/geocode?q='+encodeURIComponent(q));const d=await r.json();if(d.error)throw new Error(d.error);lat=d.lat;lon=d.lon;map.setView([lat,lon],mode==='terrain'?12:14);um();uco();msg('ok','✓ '+(d.name||q).substring(0,40))}catch(e){msg('er',e.message)}}async function preview(){const btn=$('bp');btn.disabled=true;btn.innerHTML='<div class="sp"></div>';$('bdg').textContent='...';$('bdg').style.display='inline';try{const endpoint=mode==='terrain'?'/api/terrain-preview':'/api/preview';const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pm())});const d=await r.json();if(d.error)throw new Error(d.error);$('ph').style.display='none';$('pvs').innerHTML=d.svg||d.preview;$('sseg').textContent=d.stats?.segments||d.stats?.points||'—';$('bdg').textContent='Ready';msg('ok','✓ Preview loaded')}catch(e){msg('er',e.message);$('bdg').style.display='none'}finally{btn.disabled=false;btn.innerHTML='👁 Preview'}}async function expSVG(){if(mode==='terrain'){msg('er','SVG not available for terrain');return}const btn=$('bsv');btn.disabled=true;btn.innerHTML='<div class="sp"></div>';try{const r=await fetch('/api/svg',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pm())});if(!r.ok)throw new Error((await r.json()).error);dl(await r.blob(),`map_${lat.toFixed(4)}_${lon.toFixed(4)}_${sty}.svg`);msg('ok','✓ SVG')}catch(e){msg('er',e.message)}finally{btn.disabled=false;btn.innerHTML='📄 SVG'}}async function expSTL(neg=false){const btn=$('bst');btn.disabled=true;btn.innerHTML='<div class="sp"></div>';try{const p={...pm(),negative:neg};if(mode==='terrain'){const r=await fetch('/api/terrain-stl',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});if(!r.ok)throw new Error((await r.json()).error);dl(await r.blob(),`terrain_${lat.toFixed(4)}_${lon.toFixed(4)}.stl`);msg('ok','✓ Terrain STL')}else if(p.separate_marker&&p.marker_type!=='none'){const r=await fetch('/api/stl-separate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});if(!r.ok)throw new Error((await r.json()).error);const blob=await r.blob();dl(blob,`map_${lat.toFixed(4)}_${lon.toFixed(4)}_multicolor.zip`);msg('ok','✓ ZIP with streets + marker STL')}else{const r=await fetch('/api/stl',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});if(!r.ok)throw new Error((await r.json()).error);dl(await r.blob(),`map_${lat.toFixed(4)}_${lon.toFixed(4)}_${neg?'negative':'normal'}.stl`);msg('ok','✓ STL')}}catch(e){msg('er',e.message)}finally{btn.disabled=false;btn.innerHTML='⬇ STL'}}function dl(b,n){const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=n;a.click();URL.revokeObjectURL(a.href)}function msg(t,m){const el=$('sts');el.className='sts '+t;el.textContent=m;setTimeout(()=>el.className='sts',5000)}document.addEventListener('DOMContentLoaded',init)</script></body></html>'''

@app.route('/')
def index():return render_template_string(HTML)
@app.route('/api/geocode')
def api_geo():
    q=request.args.get('q','')
    if not q:return jsonify({'error':'Missing'}),400
    r=geocode(q);return jsonify(r) if r else(jsonify({'error':'Not found'}),404)
@app.route('/api/preview',methods=['POST'])
def api_pv():
    try:
        d=request.json;gdf=fetch_streets(d['lat'],d['lon'],d['radius'],d.get('street_types'))
        svg=gen_svg(gdf,d['lat'],d['lon'],d['radius'],d['line_width'],d['target_size'],d.get('style','lines'),d.get('bg_color','#1a1a1a'),d.get('line_color','#00AE42'),d.get('marker_type','none'),d.get('marker_size',12),d.get('marker_gap',2))
        return jsonify({'svg':svg,'stats':{'segments':len(gdf)}})
    except Exception as e:return jsonify({'error':str(e)}),500
@app.route('/api/svg',methods=['POST'])
def api_svg():
    try:
        d=request.json;gdf=fetch_streets(d['lat'],d['lon'],d['radius'],d.get('street_types'))
        svg=gen_svg(gdf,d['lat'],d['lon'],d['radius'],d['line_width'],d['target_size'],d.get('style','lines'),d.get('bg_color','#1a1a1a'),d.get('line_color','#00AE42'),d.get('marker_type','none'),d.get('marker_size',12),d.get('marker_gap',2))
        buf=io.BytesIO(svg.encode());buf.seek(0);return send_file(buf,mimetype='image/svg+xml',as_attachment=True,download_name='map.svg')
    except Exception as e:return jsonify({'error':str(e)}),500
@app.route('/api/stl',methods=['POST'])
def api_stl():
    try:
        d=request.json;gdf=fetch_streets(d['lat'],d['lon'],d['radius'],d.get('street_types'))
        mesh=gen_stl(gdf,d['lat'],d['lon'],d['radius'],d['line_width'],d['target_size'],d['extrusion_height'],d.get('negative',False),d.get('marker_type','none'),d.get('marker_size',12),d.get('marker_gap',2))
        buf=io.BytesIO();mesh.export(buf,file_type='stl');buf.seek(0);return send_file(buf,mimetype='application/octet-stream',as_attachment=True,download_name='map.stl')
    except Exception as e:return jsonify({'error':str(e)}),500
@app.route('/api/stl-separate',methods=['POST'])
def api_stl_sep():
    """Export streets and marker as separate STL files in a ZIP."""
    try:
        import zipfile
        d=request.json;gdf=fetch_streets(d['lat'],d['lon'],d['radius'],d.get('street_types'))
        street_mesh, marker_mesh = gen_stl_separate(gdf,d['lat'],d['lon'],d['radius'],d['line_width'],d['target_size'],d['extrusion_height'],d.get('marker_type','none'),d.get('marker_size',12),d.get('marker_gap',2))
        
        # Create ZIP with both STLs
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            if street_mesh:
                stl_buf = io.BytesIO()
                street_mesh.export(stl_buf, file_type='stl')
                zf.writestr('streets.stl', stl_buf.getvalue())
            if marker_mesh:
                marker_buf = io.BytesIO()
                marker_mesh.export(marker_buf, file_type='stl')
                zf.writestr(f'marker_{d.get("marker_type","heart")}.stl', marker_buf.getvalue())
        
        zip_buf.seek(0)
        return send_file(zip_buf, mimetype='application/zip', as_attachment=True, download_name='map_multicolor.zip')
    except Exception as e:
        return jsonify({'error':str(e)}),500
@app.route('/api/terrain-preview',methods=['POST'])
def api_terrain_pv():
    try:
        d=request.json
        elev,lats,lons=fetch_elevation(d['lat'],d['lon'],d['radius'],resolution=50)
        size=d.get('target_size',80)
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}mm" height="{size}mm">'
        svg+=f'<rect width="100%" height="100%" fill="#1a1a1a"/>'
        res=elev.shape[0]
        elev_norm=(elev-elev.min())/(elev.max()-elev.min()+0.001)
        step=size/res
        for i in range(res):
            for j in range(res):
                v=elev_norm[i,j]
                c=int(v*255)
                color=f'rgb({c//2},{c},{c//2})'
                svg+=f'<rect x="{j*step}" y="{i*step}" width="{step+.5}" height="{step+.5}" fill="{color}"/>'
        svg+='</svg>'
        return jsonify({'svg':svg,'stats':{'points':res*res,'min_elev':f'{elev.min():.0f}m','max_elev':f'{elev.max():.0f}m'}})
    except Exception as e:return jsonify({'error':str(e)}),500
@app.route('/api/terrain-stl',methods=['POST'])
def api_terrain_stl():
    try:
        d=request.json
        res=d.get('resolution',100)
        elev,_,_=fetch_elevation(d['lat'],d['lon'],d['radius'],resolution=res)
        mesh=create_terrain_mesh(elev,d.get('target_size',80),d.get('height_scale',15))
        buf=io.BytesIO();mesh.export(buf,file_type='stl');buf.seek(0)
        return send_file(buf,mimetype='application/octet-stream',as_attachment=True,download_name='terrain.stl')
    except Exception as e:return jsonify({'error':str(e)}),500
@app.route('/api/health')
def health():return jsonify({'status':'ok','version':'2.2','author':'SebGE','features':['streets','terrain','markers']})

if __name__=='__main__':
    port=int(os.environ.get('PORT',8080))
    print(f"\n  🗺️  Map → STL v2.2 by SebGE\n  → http://localhost:{port}\n  Features: Streets + Terrain + Markers!\n")
    app.run(host='0.0.0.0',port=port,debug=os.environ.get('DEBUG','false').lower()=='true')
