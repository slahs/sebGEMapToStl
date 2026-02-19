# -*- coding: utf-8 -*-
"""SebGE Tools v2.6 - Map/Image/Print/Shadow to STL"""
import os,io,requests,base64
from flask import Flask,request,jsonify,send_file
from flask_cors import CORS
import geopandas as gpd
import numpy as np
from shapely.geometry import box,Polygon,LineString
from shapely.ops import unary_union
import trimesh
from PIL import Image, ImageFilter, ImageOps
from skimage import feature, filters, measure, exposure
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu, gaussian
from scipy.ndimage import gaussian_filter1d, uniform_filter, laplace

app=Flask(__name__)
CORS(app)

# === MARKERS ===
def create_heart(s=10):
    t=np.linspace(0,2*np.pi,80);x=16*np.sin(t)**3;y=13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    return Polygon(list(zip(x/32*s,y/32*s+s*0.1)))
def create_star(s=10):
    angles=np.linspace(-np.pi/2,3*np.pi/2,11)[:-1];coords=[]
    for i,a in enumerate(angles):r=s/2 if i%2==0 else s/5;coords.append((r*np.cos(a),r*np.sin(a)))
    return Polygon(coords)
def create_pin(s=10):
    t=np.linspace(0,2*np.pi,40);r=s/2*(1-0.3*np.cos(t));return Polygon(list(zip(r*np.sin(t),r*np.cos(t)*0.7+s*0.15)))
def create_circle(s=10):
    t=np.linspace(0,2*np.pi,50);return Polygon([(s/2*np.cos(a),s/2*np.sin(a)) for a in t])
MARKERS={'heart':create_heart,'star':create_star,'pin':create_pin,'circle':create_circle,'none':None}

STREET_TYPES={'motorway':'motorway|motorway_link','trunk':'trunk|trunk_link','primary':'primary|primary_link','secondary':'secondary|secondary_link','tertiary':'tertiary|tertiary_link','residential':'residential|living_street','unclassified':'unclassified','service':'service','pedestrian':'pedestrian','footway':'footway|steps','cycleway':'cycleway','track':'track','bridleway':'bridleway','path':'path'}

# === MAP FUNCTIONS ===
def geocode(q):
    known={'berlin':(52.52,13.405),'dusseldorf':(51.2277,6.7735),'koln':(50.9375,6.9603),'munchen':(48.1351,11.582),'hamburg':(53.5511,9.9937),'frankfurt':(50.1109,8.6821),'paris':(48.8566,2.3522),'new york':(40.7128,-74.006),'london':(51.5074,-0.1278),'tokyo':(35.6762,139.6503)}
    ql=q.lower().strip()
    for n,c in known.items():
        if n in ql:return{'lat':c[0],'lon':c[1],'name':q}
    try:
        p=ql.replace(',',' ').split()
        if len(p)>=2:
            lat,lon=float(p[0]),float(p[1])
            if -90<=lat<=90 and -180<=lon<=180:return{'lat':lat,'lon':lon,'name':f'{lat:.4f}, {lon:.4f}'}
    except:pass
    try:
        r=requests.get('https://nominatim.openstreetmap.org/search',params={'q':q,'format':'json','limit':1},headers={'User-Agent':'SebGETools/1.2'},timeout=10)
        if r.ok and r.json():d=r.json()[0];return{'lat':float(d['lat']),'lon':float(d['lon']),'name':d.get('display_name',q)[:50]}
    except:pass
    return None

def fetch_streets(lat,lon,radius,types=None):
    if types is None:types=['motorway','trunk','primary','secondary','tertiary','residential']
    tags=[STREET_TYPES[t] for t in types if t in STREET_TYPES]
    if not tags:tags=['residential']
    regex='|'.join(tags);bbox=radius/111000
    q=f'[out:json][timeout:90];(way["highway"~"^({regex})$"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5}););out body;>;out skel qt;'
    servers = ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter','https://maps.mail.ru/osm/tools/overpass/api/interpreter']
    for server in servers:
        try:
            r=requests.post(server,data={'data':q},timeout=90,headers={'User-Agent':'SebGE-Tools/1.4'})
            if r.ok and r.text.strip().startswith('{'):
                data=r.json()
                nodes={e['id']:(e['lon'],e['lat']) for e in data.get('elements',[]) if e['type']=='node'}
                lines=[]
                for e in data.get('elements',[]):
                    if e['type']=='way' and 'nodes' in e:
                        coords=[nodes[n] for n in e['nodes'] if n in nodes]
                        if len(coords)>=2:lines.append(LineString(coords))
                if lines:return gpd.GeoDataFrame(geometry=lines,crs='EPSG:4326')
        except:continue
    raise ValueError(f"Could not fetch street data")

def fetch_buildings(lat,lon,radius):
    bbox=radius/111000;q=f'[out:json][timeout:90];way["building"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5});out body;>;out skel qt;'
    servers = ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter']
    for server in servers:
        try:
            r=requests.post(server,data={'data':q},timeout=90,headers={'User-Agent':'SebGE-Tools/1.4'})
            if not r.ok or not r.text.strip().startswith('{'):continue
            data=r.json();nodes={e['id']:(e['lon'],e['lat']) for e in data.get('elements',[]) if e['type']=='node'}
            buildings=[]
            for e in data.get('elements',[]):
                if e['type']=='way' and 'nodes' in e:
                    coords=[nodes[n] for n in e['nodes'] if n in nodes]
                    if len(coords)>=4:
                        try:
                            poly=Polygon(coords)
                            if poly.is_valid and poly.area>0:buildings.append({'geometry':poly,'height':10})
                        except:pass
            if buildings:return buildings
        except:continue
    return []

def fetch_elevation(lat,lon,radius,res=80):
    dl=radius/111320;dlo=dl/np.cos(np.radians(lat))
    try:
        locs=[{"latitude":la,"longitude":lo} for la in np.linspace(lat-dl,lat+dl,8) for lo in np.linspace(lon-dlo,lon+dlo,8)]
        r=requests.post("https://api.open-elevation.com/api/v1/lookup",json={"locations":locs},timeout=20)
        if r.ok:
            from scipy.ndimage import zoom
            grid=np.array([p["elevation"] for p in r.json()["results"]]).reshape(8,8)
            return zoom(grid,res/8,order=3)
    except:pass
    np.random.seed(int(abs(lat*1000+lon*100)));x,y=np.meshgrid(np.linspace(0,4,res),np.linspace(0,4,res))
    t=np.zeros((res,res))
    for f,a in [(1,400),(2,200),(4,100),(8,50)]:t+=a*(np.sin(f*x+np.random.rand()*10)*np.cos(f*y+np.random.rand()*10)+1)/2
    return 200+(t-t.min())/(t.max()-t.min())*1500

def create_streets_mesh(gdf,lat,lon,radius,size,height,lw,marker_type='none',marker_size=10,marker_gap=0):
    gp=gdf.to_crs(gdf.estimate_utm_crs())
    ctr=gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon],[lat]),crs='EPSG:4326').to_crs(gp.crs).geometry.iloc[0]
    bb=box(ctr.x-radius,ctr.y-radius,ctr.x+radius,ctr.y+radius);cl=gp.clip(bb)
    if cl.empty:raise ValueError("No data")
    sc=size/(2*radius);buf=lw/sc/2
    marker_poly=gap_poly=None
    if marker_type!='none' and marker_type in MARKERS and MARKERS[marker_type]:
        marker_poly=MARKERS[marker_type](marker_size)
        if marker_gap>0:gap_poly=marker_poly.buffer(marker_gap)
    uni=unary_union(cl.geometry.buffer(buf,cap_style=2,join_style=2)).simplify(0.3)
    def norm(p):
        c=np.array(p.exterior.coords);c[:,0]=(c[:,0]-ctr.x)*sc;c[:,1]=(c[:,1]-ctr.y)*sc
        return Polygon(c.tolist(),[((np.array(i.coords)-[ctr.x,ctr.y])*sc).tolist() for i in p.interiors])
    polys=[norm(p) for p in(uni.geoms if hasattr(uni,'geoms')else[uni]) if not p.is_empty]
    streets=unary_union(polys) if len(polys)>1 else(polys[0] if polys else None)
    if not streets:raise ValueError("No streets")
    if gap_poly:streets=streets.difference(gap_poly)
    meshes=[]
    for p in(streets.geoms if hasattr(streets,'geoms')else[streets]):
        if p.is_empty or p.area<0.1:continue
        try:meshes.append(trimesh.creation.extrude_polygon(p,height))
        except:pass
    if marker_poly:
        try:meshes.append(trimesh.creation.extrude_polygon(marker_poly,height))
        except:pass
    if not meshes:raise ValueError("No mesh")
    result=trimesh.util.concatenate(meshes);result.fill_holes();result.fix_normals()
    return result

def create_city_mesh(lat,lon,radius,size,street_h,lw,building_scale,base_h):
    gdf=fetch_streets(lat,lon,radius);buildings=fetch_buildings(lat,lon,radius)
    gp=gdf.to_crs(gdf.estimate_utm_crs());utm=gp.crs
    ctr=gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon],[lat]),crs='EPSG:4326').to_crs(utm).geometry.iloc[0]
    sc=size/(2*radius);meshes=[trimesh.creation.extrude_polygon(box(-size/2,-size/2,size/2,size/2),base_h)]
    bb=box(ctr.x-radius,ctr.y-radius,ctr.x+radius,ctr.y+radius);cl=gp.clip(bb)
    if not cl.empty:
        uni=unary_union(cl.geometry.buffer(lw/sc,cap_style=2)).simplify(0.3)
        for p in(uni.geoms if hasattr(uni,'geoms')else[uni]):
            if p.is_empty or p.area<1:continue
            c=np.array(p.exterior.coords);c[:,0]=(c[:,0]-ctr.x)*sc;c[:,1]=(c[:,1]-ctr.y)*sc
            try:m=trimesh.creation.extrude_polygon(Polygon(c.tolist()),street_h);m.vertices[:,2]+=base_h;meshes.append(m)
            except:pass
    for b in buildings:
        try:
            bg=gpd.GeoDataFrame(geometry=[b['geometry']],crs='EPSG:4326').to_crs(utm).geometry.iloc[0]
            if bg.distance(ctr)>radius*1.2:continue
            c=np.array(bg.exterior.coords);c[:,0]=(c[:,0]-ctr.x)*sc;c[:,1]=(c[:,1]-ctr.y)*sc
            bp=Polygon(c.tolist()).intersection(box(-size/2,-size/2,size/2,size/2))
            if bp.is_empty or bp.area<0.5:continue
            bh=max(b['height']*building_scale*sc,0.5)
            m=trimesh.creation.extrude_polygon(bp if bp.geom_type=='Polygon' else list(bp.geoms)[0],bh)
            m.vertices[:,2]+=base_h+street_h;meshes.append(m)
        except:pass
    result=trimesh.util.concatenate(meshes);result.fill_holes();result.fix_normals()
    return result

def create_terrain_mesh(lat,lon,radius,size,height_scale,base_h):
    elev=fetch_elevation(lat,lon,radius,80);res=elev.shape[0]
    e_range=elev.max()-elev.min() or 1;norm=(elev-elev.min())/e_range*height_scale
    verts=[];faces=[];step=size/(res-1);half=size/2
    for i in range(res):
        for j in range(res):verts.append([-half+j*step,-half+i*step,base_h+norm[i,j]])
    for i in range(res):
        for j in range(res):verts.append([-half+j*step,-half+i*step,0])
    n=res;off=res*res
    for i in range(res-1):
        for j in range(res-1):
            v0,v1,v2,v3=i*n+j,i*n+j+1,(i+1)*n+j,(i+1)*n+j+1
            faces+=[[v0,v2,v1],[v1,v2,v3],[off+v0,off+v1,off+v2],[off+v1,off+v3,off+v2]]
    for j in range(res-1):faces+=[[j,j+1,off+j],[j+1,off+j+1,off+j]]
    for j in range(res-1):t1,t2=(res-1)*n+j,(res-1)*n+j+1;b1,b2=off+t1,off+t2;faces+=[[t1,b1,t2],[t2,b1,b2]]
    for i in range(res-1):t1,t2=i*n,(i+1)*n;b1,b2=off+t1,off+t2;faces+=[[t1,b1,t2],[t2,b1,b2]]
    for i in range(res-1):t1,t2=i*n+res-1,(i+1)*n+res-1;b1,b2=off+t1,off+t2;faces+=[[t1,t2,b1],[t2,b2,b1]]
    mesh=trimesh.Trimesh(vertices=verts,faces=faces);mesh.fix_normals()
    return mesh

def create_terrain_mesh_with_sampler(lat,lon,radius,size,height_scale,base_h,res=80):
    elev = fetch_elevation(lat,lon,radius,res)
    res = elev.shape[0]
    e_range = elev.max() - elev.min() or 1.0
    norm = (elev - elev.min())/e_range * float(height_scale)
    step = size/(res-1);half = size/2.0
    verts=[];faces=[]
    for i in range(res):
        for j in range(res):verts.append([-half + j*step, -half + i*step, base_h + norm[i,j]])
    for i in range(res):
        for j in range(res):verts.append([-half + j*step, -half + i*step, 0.0])
    n=res;off=res*res
    for i in range(res-1):
        for j in range(res-1):
            v0,v1,v2,v3=i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1
            faces += [[v0,v2,v1],[v1,v2,v3],[off+v0,off+v1,off+v2],[off+v1,off+v3,off+v2]]
    for j in range(res-1):faces += [[j, j+1, off+j], [j+1, off+j+1, off+j]]
    for j in range(res-1):t1,t2=(res-1)*n+j, (res-1)*n+j+1;b1,b2=off+t1, off+t2;faces += [[t1,b1,t2],[t2,b1,b2]]
    for i in range(res-1):t1,t2=i*n, (i+1)*n;b1,b2=off+t1, off+t2;faces += [[t1,b1,t2],[t2,b1,b2]]
    for i in range(res-1):t1,t2=i*n+res-1, (i+1)*n+res-1;b1,b2=off+t1, off+t2;faces += [[t1,t2,b1],[t2,b2,b1]]
    terrain = trimesh.Trimesh(vertices=verts,faces=faces);terrain.fix_normals()
    def sample_z(x,y):
        fx = (x + half)/step;fy = (y + half)/step
        if fx < 0: fx = 0.0
        if fy < 0: fy = 0.0
        if fx > res-1: fx = float(res-1)
        if fy > res-1: fy = float(res-1)
        x0 = int(np.floor(fx)); x1 = min(x0+1, res-1)
        y0 = int(np.floor(fy)); y1 = min(y0+1, res-1)
        tx = fx - x0; ty = fy - y0
        z00 = norm[y0,x0]; z10 = norm[y0,x1];z01 = norm[y1,x0]; z11 = norm[y1,x1]
        z0 = z00*(1-tx) + z10*tx;z1 = z01*(1-tx) + z11*tx
        return float(base_h + (z0*(1-ty) + z1*ty))
    return terrain, sample_z

def _polygons_from_streets(gdf, lat, lon, radius, size, line_width_mm, street_types=None):
    gp = gdf.to_crs(gdf.estimate_utm_crs())
    ctr = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon],[lat]),crs='EPSG:4326').to_crs(gp.crs).geometry.iloc[0]
    bb = box(ctr.x-radius, ctr.y-radius, ctr.x+radius, ctr.y+radius);cl = gp.clip(bb)
    if cl.empty:return None, gp.crs, ctr
    sc = size/(2*radius);buf = (line_width_mm/sc)/2.0
    uni = unary_union(cl.geometry.buffer(buf, cap_style=2, join_style=2)).simplify(0.15)
    def norm_poly(p):
        c = np.array(p.exterior.coords);c[:,0] = (c[:,0]-ctr.x)*sc;c[:,1] = (c[:,1]-ctr.y)*sc
        holes=[]
        for ring in p.interiors:
            rc = np.array(ring.coords);rc[:,0] = (rc[:,0]-ctr.x)*sc;rc[:,1] = (rc[:,1]-ctr.y)*sc
            holes.append(rc.tolist())
        return Polygon(c.tolist(), holes)
    polys = [norm_poly(p) for p in (uni.geoms if hasattr(uni,'geoms') else [uni]) if (not p.is_empty)]
    if not polys:return None, gp.crs, ctr
    streets = unary_union(polys) if len(polys)>1 else polys[0]
    return streets, gp.crs, ctr

def _building_polygons(lat, lon, radius, size, utm_crs, ctr_utm, buildings):
    sc = size/(2*radius);box_clip = box(-size/2, -size/2, size/2, size/2);out=[]
    for b in buildings:
        try:
            bg = gpd.GeoDataFrame(geometry=[b['geometry']], crs='EPSG:4326').to_crs(utm_crs).geometry.iloc[0]
            if bg.is_empty or not hasattr(bg,'exterior'):continue
            if bg.distance(ctr_utm) > radius*1.2:continue
            c = np.array(bg.exterior.coords);c[:,0] = (c[:,0]-ctr_utm.x)*sc;c[:,1] = (c[:,1]-ctr_utm.y)*sc
            bp = Polygon(c.tolist()).intersection(box_clip)
            if bp.is_empty or bp.area < 0.6:continue
            out.append(bp if bp.geom_type=='Polygon' else list(bp.geoms)[0])
        except:pass
    return out

def _drape_mesh_vertices(mesh, sample_z_fn, z_bias=0.0):
    v = mesh.vertices.copy();zmin = v[:,2].min()
    base = np.array([sample_z_fn(x,y) for x,y,_ in v], dtype=float) + float(z_bias)
    v[:,2] = base + (v[:,2] - zmin);mesh.vertices = v;mesh.fix_normals()
    return mesh

def _place_building(mesh, sample_z_fn, foundation_h, building_h):
    v = mesh.vertices.copy();zmin = v[:,2].min();zmax = v[:,2].max()
    base = np.array([sample_z_fn(x,y) for x,y,_ in v], dtype=float)
    base_mean = float(np.mean(base))
    bottom_mask = np.isclose(v[:,2], zmin);top_mask = np.isclose(v[:,2], zmax)
    v[bottom_mask,2] = base[bottom_mask] + foundation_h
    v[top_mask,2] = base_mean + foundation_h + building_h
    mid_mask = ~(bottom_mask | top_mask)
    if np.any(mid_mask):
        t = (v[mid_mask,2] - zmin) / (zmax - zmin + 1e-9)
        v[mid_mask,2] = (base[mid_mask] + foundation_h) * (1-t) + (base_mean + foundation_h + building_h) * t
    mesh.vertices = v;mesh.fix_normals()
    return mesh

def create_composite_mesh(lat, lon, radius, size, road_h, line_width_mm=1.2,terrain_scale=None, base_h=1.5,foundation_h=0.8, building_scale=4.0,street_types=None):
    if terrain_scale is None:terrain_scale = float(road_h) * 5.0
    terrain, sample_z = create_terrain_mesh_with_sampler(lat, lon, radius, size, terrain_scale, base_h, res=80)
    gdf = fetch_streets(lat, lon, radius, street_types)
    streets, utm_crs, ctr_utm = _polygons_from_streets(gdf, lat, lon, radius, size, line_width_mm, street_types)
    meshes = [terrain]
    if streets:
        for p in (streets.geoms if hasattr(streets,'geoms') else [streets]):
            if p.is_empty or p.area < 0.15:continue
            try:
                m = trimesh.creation.extrude_polygon(p, float(road_h))
                m = _drape_mesh_vertices(m, sample_z, z_bias=0.0);meshes.append(m)
            except:pass
    bld = fetch_buildings(lat, lon, radius)
    b_polys = _building_polygons(lat, lon, radius, size, utm_crs, ctr_utm, bld) if utm_crs else []
    for bp in b_polys:
        try:
            f = trimesh.creation.extrude_polygon(bp, float(foundation_h))
            f = _drape_mesh_vertices(f, sample_z, z_bias=0.0);meshes.append(f)
            bh_mm = max(10.0 * float(building_scale) * (size/(2*radius)), 1.0)
            bmesh = trimesh.creation.extrude_polygon(bp, float(bh_mm))
            bmesh = _place_building(bmesh, sample_z, float(foundation_h), float(bh_mm));meshes.append(bmesh)
        except:pass
    result = trimesh.util.concatenate(meshes);result.fill_holes();result.fix_normals()
    return result

# === IMAGE TO SVG ===
def smooth_contour(contour, sigma):
    """Gaussian smoothing of contour points"""
    if sigma <= 0 or len(contour) < 5:
        return contour
    smoothed = np.zeros_like(contour)
    smoothed[:, 0] = gaussian_filter1d(contour[:, 0], sigma=sigma, mode='wrap')
    smoothed[:, 1] = gaussian_filter1d(contour[:, 1], sigma=sigma, mode='wrap')
    return smoothed

def collinear_simplify(points, angle_threshold_deg=3):
    """
    Removes points that lie on a straight line.
    Only keeps points where direction changes by more than angle_threshold.
    This is KEY for clean CAD import - straight edges become single lines!
    
    angle_threshold_deg: 
        1-2 = very detailed (keeps slight curves)
        3-5 = good for most cases
        8-15 = aggressive (only sharp corners)
    """
    if len(points) < 3:
        return points
    
    threshold_rad = np.radians(angle_threshold_deg)
    keep = [0]  # Always keep first point
    
    for i in range(1, len(points) - 1):
        # Vector from last kept point to current
        v1 = points[i] - points[keep[-1]]
        # Vector from current to next
        v2 = points[i + 1] - points[i]
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 < 0.001 or len2 < 0.001:
            continue
        
        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Keep point if angle is significant
        if angle > threshold_rad:
            keep.append(i)
    
    keep.append(len(points) - 1)  # Always keep last point
    return points[np.array(keep)]

def simplify_contour_for_cad(contour, simplify_level):
    """
    Two-stage simplification optimized for CAD:
    1. First reduce point count roughly (nth point)
    2. Then remove collinear points (straight lines → 2 points)
    
    simplify_level: 1-10 slider value
        1 = detailed
        5 = balanced  
        10 = very simplified
    """
    if len(contour) < 4:
        return contour
    
    # Stage 1: Rough reduction if very detailed
    if len(contour) > 200:
        step = max(1, len(contour) // 200)
        contour = contour[::step]
    
    # Stage 2: Collinear simplification
    # Map simplify 1-10 to angle threshold 1-15 degrees
    angle_threshold = 1 + (simplify_level - 1) * 1.5  # 1° to 14.5°
    
    simplified = collinear_simplify(contour, angle_threshold)
    
    # Ensure closed contour
    if len(simplified) > 2 and not np.allclose(simplified[0], simplified[-1], atol=0.5):
        simplified = np.vstack([simplified, simplified[0]])
    
    return simplified

def image_to_svg(img_data, threshold=128, blur=1, simplify=2, smooth=0, invert=False, mode='outline', single_line=False):
    """Convert image to SVG - optimized for CAD import (Fusion 360, etc.)"""
    img = Image.open(io.BytesIO(img_data))
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    w, h = img.size
    gray = img.convert('L')
    
    if blur > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur))
    if invert:
        gray = ImageOps.invert(gray)
    
    img_array = np.array(gray)
    
    # Single line mode - skeleton tracing
    if single_line or mode == 'centerline':
        binary = img_array < threshold
        skeleton = skeletonize(binary)
        points = np.argwhere(skeleton)
        
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        svg += '<rect width="100%" height="100%" fill="white"/>'
        
        if len(points) > 0:
            # Simplify skeleton points
            step = max(1, simplify)
            points = points[::step]
            
            if len(points) > 1:
                d = f"M {points[0][1]:.1f},{points[0][0]:.1f}"
                for p in points[1:]:
                    d += f" L {p[1]:.1f},{p[0]:.1f}"
                svg += f'<path d="{d}" fill="none" stroke="black" stroke-width="1"/>'
        
        svg += '</svg>'
        return svg, 1 if len(points) > 0 else 0, w, h
    
    # Outline & Threshold mode - contour tracing
    binary = (img_array < threshold).astype(np.uint8) * 255
    contours = measure.find_contours(binary, 0.5)
    
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
    svg += '<rect width="100%" height="100%" fill="white"/>'
    
    paths = []
    total_points = 0
    
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Apply smoothing first (removes pixel-level noise)
        if smooth > 0:
            contour = smooth_contour(contour, sigma=smooth * 2)
        
        # CAD-optimized simplification
        contour = simplify_contour_for_cad(contour, simplify)
        
        if len(contour) < 3:
            continue
        
        total_points += len(contour)
        
        # Build path (contour is y,x format)
        d = f"M {contour[0][1]:.1f},{contour[0][0]:.1f}"
        for p in contour[1:]:
            d += f" L {p[1]:.1f},{p[0]:.1f}"
        d += " Z"
        paths.append(d)
    
    if paths:
        svg += f'<path d="{" ".join(paths)}" fill="none" stroke="black" stroke-width="1"/>'
    
    svg += '</svg>'
    return svg, len(paths), w, h

def image_to_filled_svg(img_data, threshold=128, blur=1, simplify=2, smooth=0, invert=False):
    """Convert image to filled SVG - optimized for CAD import"""
    img = Image.open(io.BytesIO(img_data))
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    w, h = img.size
    gray = img.convert('L')
    
    if blur > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur))
    if invert:
        gray = ImageOps.invert(gray)
    
    img_array = np.array(gray)
    binary = (img_array < threshold).astype(np.uint8) * 255
    contours = measure.find_contours(binary, 0.5)
    
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
    svg += '<rect width="100%" height="100%" fill="white"/>'
    
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Smoothing
        if smooth > 0:
            contour = smooth_contour(contour, sigma=smooth * 2)
        
        # CAD-optimized simplification  
        contour = simplify_contour_for_cad(contour, simplify)
        
        if len(contour) < 3:
            continue
        
        # Build path
        d = f"M {contour[0][1]:.1f},{contour[0][0]:.1f}"
        for p in contour[1:]:
            d += f" L {p[1]:.1f},{p[0]:.1f}"
        d += " Z"
        paths.append(d)
    
    if paths:
        svg += f'<path d="{" ".join(paths)}" fill="black" stroke="none" fill-rule="evenodd"/>'
    
    svg += '</svg>'
    return svg, len(paths), w, h

# === PRINT TO STL ===
def analyze_print_image(img_data):
    img = Image.open(io.BytesIO(img_data))
    if img.mode == 'RGBA':bg = Image.new('RGB', img.size, (255, 255, 255));bg.paste(img, mask=img.split()[3]);img = bg
    elif img.mode != 'RGB':img = img.convert('RGB')
    gray = np.array(img.convert('L'));warnings = []
    mean_brightness = np.mean(gray)
    if mean_brightness < 60:warnings.append("Image is very dark")
    elif mean_brightness > 220:warnings.append("Image is very bright")
    contrast = np.std(gray)
    if contrast < 30:warnings.append("Low contrast")
    quality_score = 100
    if mean_brightness < 60 or mean_brightness > 220:quality_score -= 25
    if contrast < 30:quality_score -= 20
    return {'width': img.size[0],'height': img.size[1],'quality_score': max(0, quality_score),'warnings': warnings}

def process_print_image(img_data, enhance_contrast=1.5, denoise=2):
    img = Image.open(io.BytesIO(img_data))
    if img.mode == 'RGBA':bg = Image.new('RGB', img.size, (255, 255, 255));bg.paste(img, mask=img.split()[3]);img = bg
    elif img.mode != 'RGB':img = img.convert('RGB')
    gray = np.array(img.convert('L')).astype(float)
    if denoise > 0:gray = gaussian(gray, sigma=denoise, preserve_range=True)
    h, w = gray.shape;corner_size = max(10, min(h, w) // 10)
    corners = [gray[:corner_size, :corner_size], gray[:corner_size, -corner_size:], gray[-corner_size:, :corner_size], gray[-corner_size:, -corner_size:]]
    bg_value = np.median([np.median(c) for c in corners])
    if bg_value > 128:
        p_dark = np.percentile(gray, 2);gray = np.clip((gray - p_dark) / (bg_value - p_dark + 1e-8) * 255, 0, 255)
    else:
        p_light = np.percentile(gray, 98);gray = 255 - np.clip((gray - bg_value) / (p_light - bg_value + 1e-8) * 255, 0, 255)
    if enhance_contrast > 1:mid = 128;gray = mid + (gray - mid) * enhance_contrast;gray = np.clip(gray, 0, 255)
    p2, p98 = np.percentile(gray, (1, 99));gray = exposure.rescale_intensity(gray, in_range=(p2, p98), out_range=(0, 255))
    return gray.astype(np.uint8)

def create_hueforge_preview(gray, num_colors=2):
    h, w = gray.shape;preview = np.zeros((h, w, 3), dtype=np.uint8)
    if num_colors == 2:
        preview[gray >= 128] = [255, 255, 255];preview[gray < 128] = [30, 30, 30]
    else:
        preview[gray >= 170] = [255, 255, 255];preview[(gray >= 85) & (gray < 170)] = [120, 120, 120];preview[gray < 85] = [30, 30, 30]
    return preview

def create_relief_stl(gray, width_mm=100, total_height_mm=2.4, base_height_mm=0.6, border_mm=2):
    h, w = gray.shape;aspect = h / w;width = width_mm;height = width * aspect
    if border_mm > 0:
        padded = np.pad(gray, pad_width=int(border_mm * w / width_mm), mode='constant', constant_values=255)
        gray = padded;h, w = gray.shape;height = width * (h / w)
    max_res = 400
    if w > max_res or h > max_res:
        scale = max_res / max(w, h);new_w, new_h = int(w * scale), int(h * scale)
        img = Image.fromarray(gray);img = img.resize((new_w, new_h), Image.Resampling.LANCZOS);gray = np.array(img);h, w = gray.shape
    relief_height = total_height_mm - base_height_mm;normalized = 1.0 - (gray.astype(float) / 255.0);heights = base_height_mm + normalized * relief_height
    step_x = width / (w - 1);step_y = height / (h - 1);vertices = [];faces = []
    for i in range(h):
        for j in range(w):vertices.append([j * step_x, (h - 1 - i) * step_y, heights[i, j]])
    for i in range(h):
        for j in range(w):vertices.append([j * step_x, (h - 1 - i) * step_y, 0])
    n = w;off = h * w
    for i in range(h - 1):
        for j in range(w - 1):
            v0, v1, v2, v3 = i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1
            faces.append([v0, v2, v1]);faces.append([v1, v2, v3])
    for i in range(h - 1):
        for j in range(w - 1):
            v0, v1, v2, v3 = off+i*n+j, off+i*n+j+1, off+(i+1)*n+j, off+(i+1)*n+j+1
            faces.append([v0, v1, v2]);faces.append([v1, v3, v2])
    for j in range(w - 1):faces.append([j, j+1, off+j]);faces.append([j+1, off+j+1, off+j])
    for j in range(w - 1):t0, t1 = (h-1)*n+j, (h-1)*n+j+1;b0, b1 = off+(h-1)*n+j, off+(h-1)*n+j+1;faces.append([t0, b0, t1]);faces.append([t1, b0, b1])
    for i in range(h - 1):t0, t1 = i*n, (i+1)*n;b0, b1 = off+i*n, off+(i+1)*n;faces.append([t0, b0, t1]);faces.append([t1, b0, b1])
    for i in range(h - 1):t0, t1 = i*n+w-1, (i+1)*n+w-1;b0, b1 = off+i*n+w-1, off+(i+1)*n+w-1;faces.append([t0, t1, b0]);faces.append([t1, b1, b0])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces);mesh.fix_normals()
    return mesh, width, height

def calculate_print_instructions(total_height_mm, base_height_mm, layer_height_mm, num_colors):
    total_layers = int(round(total_height_mm / layer_height_mm));base_layers = int(round(base_height_mm / layer_height_mm))
    if num_colors == 2:
        return {'total_layers': total_layers, 'summary': f"Layer Height: {layer_height_mm}mm | Total: {total_height_mm}mm | Layers: {total_layers}\n\n1. Start WHITE\n2. Change BLACK at Layer {base_layers} (Z={base_height_mm}mm)"}
    else:
        relief = total_height_mm - base_height_mm;g_layer = int(round((base_height_mm + relief * 0.33) / layer_height_mm));b_layer = int(round((base_height_mm + relief * 0.66) / layer_height_mm))
        return {'total_layers': total_layers, 'summary': f"Layer Height: {layer_height_mm}mm | Total: {total_height_mm}mm | Layers: {total_layers}\n\n1. Start WHITE\n2. GRAY at Layer {g_layer}\n3. BLACK at Layer {b_layer}"}

# === HTML TEMPLATES ===
LANDING="""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SebGE Tools - 3D Print & Design Tools</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#fff;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 20px}
.header{text-align:center;margin-bottom:40px}.logo-box{display:inline-flex;align-items:center;gap:16px;margin-bottom:16px}.logo-icon{width:72px;height:72px;background:linear-gradient(135deg,#00AE42,#008F36);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:36px;box-shadow:0 0 30px rgba(0,174,66,0.3)}.logo-text h1{font-size:36px;font-weight:700;letter-spacing:-1px}.logo-text .sub{color:#00AE42;font-size:12px;font-weight:600;letter-spacing:2px;margin-top:4px}
.creator{display:inline-flex;align-items:center;gap:12px;background:rgba(0,174,66,0.1);border:1px solid rgba(0,174,66,0.3);border-radius:50px;padding:8px 20px 8px 8px;margin-top:16px;text-decoration:none;color:#fff;transition:all .2s}.creator:hover{background:rgba(0,174,66,0.2);border-color:#00AE42}.creator-avatar{width:36px;height:36px;background:#00AE42;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px}.creator-info{text-align:left}.creator-name{font-size:13px;font-weight:600;color:#00AE42}.creator-link{font-size:11px;color:#888}
.tools{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;max-width:900px;width:100%;margin-bottom:40px}.tool{background:#151515;border:1px solid #2a2a2a;border-radius:16px;padding:32px 24px;text-decoration:none;color:#fff;transition:all .2s;position:relative;overflow:hidden}.tool:hover{border-color:#00AE42;transform:translateY(-4px);box-shadow:0 10px 40px rgba(0,0,0,0.3)}.tool-badge{position:absolute;top:16px;right:16px;background:#00AE42;color:#fff;font-size:9px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.5px}.tool-badge.vector{background:#9333EA}.tool-icon{font-size:48px;margin-bottom:16px;display:block}.tool h3{font-size:20px;font-weight:700;margin-bottom:8px}.tool p{font-size:13px;color:#888;line-height:1.5}
.support{background:linear-gradient(135deg,#1a1a1a,#151515);border:1px solid #2a2a2a;border-radius:16px;padding:24px 32px;max-width:900px;width:100%;display:flex;align-items:center;gap:24px;margin-bottom:30px}.support-icon{font-size:40px;flex-shrink:0}.support-text{flex:1}.support-text h4{font-size:15px;font-weight:600;margin-bottom:6px;color:#fff}.support-text p{font-size:12px;color:#888;line-height:1.5}.support-btn{background:#00AE42;color:#fff;text-decoration:none;padding:12px 24px;border-radius:8px;font-size:13px;font-weight:600;white-space:nowrap;transition:all .2s}.support-btn:hover{background:#00C94B;transform:scale(1.05)}.support-price{text-align:center;margin-right:8px}.support-price .amount{font-size:24px;font-weight:700;color:#00AE42}.support-price .period{font-size:10px;color:#888}
.footer{text-align:center;font-size:12px;color:#666;margin-top:auto;padding-top:20px}.footer a{color:#00AE42;text-decoration:none}.footer span{margin:0 8px}
@media(max-width:700px){.logo-box{flex-direction:column;gap:12px}.logo-text{text-align:center}.logo-text h1{font-size:28px}.tools{grid-template-columns:1fr}.support{flex-direction:column;text-align:center;gap:16px}.support-price{margin:0}}
</style></head><body>
<div class="header"><div class="logo-box"><div class="logo-icon">🛠️</div><div class="logo-text"><h1>SebGE Tools</h1><div class="sub">3D PRINT & DESIGN TOOLS</div></div></div>
<a href="https://makerworld.com/de/@SebGE" target="_blank" class="creator"><div class="creator-avatar">S</div><div class="creator-info"><div class="creator-name">Created by SebGE</div><div class="creator-link">MakerWorld Profile →</div></div></a></div>
<div class="tools">
<a href="/map" class="tool"><span class="tool-badge">3D PRINT</span><span class="tool-icon">🗺️</span><h3>Map → STL</h3><p>Create 3D printable maps from any location.</p></a>
<a href="/image" class="tool"><span class="tool-badge vector">VECTOR</span><span class="tool-icon">🎨</span><h3>Image → SVG</h3><p>Convert images to vector graphics for CAD.</p></a>
<a href="/print" class="tool"><span class="tool-badge">3D PRINT</span><span class="tool-icon">🖼️</span><h3>HueForge Print</h3><p>Create multi-color lithophane STLs from photos.</p></a>
<a href="/shadow" class="tool"><span class="tool-badge">SHADOW</span><span class="tool-icon">👪</span><h3>Shadow Maker</h3><p>Turn silhouette PNGs into STL for shadow boxes.</p></a>
</div>
<div class="support"><div class="support-icon">❤️</div><div class="support-text"><h4>Support this Project</h4><p>These tools are free! A $3/month supporter membership helps cover costs.</p></div><div class="support-price"><div class="amount">$3</div><div class="period">/month</div></div><a href="https://makerworld.com/de/@SebGE" target="_blank" class="support-btn">Support on MakerWorld</a></div>
<div class="footer">Made with ❤️ by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div>
</body></html>"""

IMAGE_HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Image to SVG | SebGE Tools</title>
<style>*{box-sizing:border-box;margin:0;padding:0}:root{--g:#00AE42;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}body{font-family:system-ui;background:var(--bg);color:var(--t);min-height:100vh}.app{display:grid;grid-template-columns:260px 1fr 200px;height:100vh}.panel{background:var(--c);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:8px}.back{color:var(--t2);text-decoration:none;font-size:10px}.back:hover{color:var(--g)}.logo{display:flex;align-items:center;gap:8px;padding-bottom:8px;border-bottom:1px solid var(--br)}.logo-icon{width:32px;height:32px;background:var(--g);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px}.logo h1{font-size:13px}.logo small{font-size:7px;color:var(--g)}.sec{background:var(--c2);border-radius:6px;padding:10px}.sec-title{font-size:8px;color:var(--g);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-weight:600}.upload{border:2px dashed var(--br);border-radius:6px;padding:20px 10px;text-align:center;cursor:pointer}.upload:hover{border-color:var(--g)}.upload p{font-size:10px;color:var(--t2)}.upload.has-img{padding:8px}.upload img{max-width:100%;max-height:100px;border-radius:4px}#file{display:none}.modes{display:grid;grid-template-columns:1fr 1fr;gap:4px}.mode{padding:8px 4px;background:var(--c);border:2px solid var(--br);border-radius:4px;cursor:pointer;text-align:center;font-size:9px}.mode:hover,.mode.active{border-color:var(--g)}.slider{margin-bottom:4px}.slider-head{display:flex;justify-content:space-between;margin-bottom:2px}.slider label{font-size:9px;color:var(--t2)}.slider .val{font-size:9px;color:var(--g);font-family:monospace}.slider input{width:100%;height:4px;background:var(--c);border-radius:2px;-webkit-appearance:none}.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--g);border-radius:50%;cursor:pointer}.check{display:flex;align-items:center;gap:6px;font-size:10px;cursor:pointer;margin-top:6px}.check input{width:14px;height:14px;accent-color:var(--g)}.btn{padding:10px;border:none;border-radius:6px;font-size:10px;font-weight:600;cursor:pointer;width:100%;margin-bottom:4px}.btn-primary{background:var(--g);color:#fff}.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}.btn:disabled{opacity:.4;cursor:not-allowed}.preview-panel{background:#fff;display:flex;align-items:center;justify-content:center;padding:20px}.preview-panel img,.preview-panel svg{max-width:100%;max-height:80vh}.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px}.stat{background:var(--c2);border-radius:4px;padding:6px;text-align:center}.stat label{font-size:7px;color:var(--t2);text-transform:uppercase}.stat .v{font-size:11px;color:var(--g);font-family:monospace}.status{padding:6px;border-radius:4px;font-size:9px;display:none;text-align:center}.status.error{display:block;background:rgba(239,68,68,.1);color:#ef4444}.status.success{display:block;background:rgba(0,174,66,.1);color:var(--g)}.footer{margin-top:auto;text-align:center;font-size:8px;color:var(--t2);padding-top:8px;border-top:1px solid var(--br)}.footer a{color:var(--g)}@media(max-width:800px){.app{display:block;height:auto}.panel,.preview-panel{min-height:50vh}}</style></head><body>
<div class="app"><div class="panel active" id="panelSettings"><a href="/" class="back">← Back</a><div class="logo"><div class="logo-icon">I</div><div><h1>Image → SVG</h1><small>VECTORIZE IMAGES</small></div></div><div class="upload" id="up" onclick="document.getElementById('file').click()"><p>Drop image here<br>or tap to select</p></div><input type="file" id="file" accept="image/*"><div class="sec"><div class="sec-title">Mode</div><div class="modes"><div class="mode active" data-m="outline" onclick="setMode('outline')">Outline</div><div class="mode" data-m="filled" onclick="setMode('filled')">Filled</div><div class="mode" data-m="threshold" onclick="setMode('threshold')">Threshold</div><div class="mode" data-m="centerline" onclick="setMode('centerline')">Single Line</div></div></div><div class="sec"><div class="sec-title">Settings</div><div class="slider"><div class="slider-head"><label>Threshold</label><span class="val" id="thV">128</span></div><input type="range" id="th" min="10" max="245" value="128" oninput="$('thV').textContent=this.value"></div><div class="slider"><div class="slider-head"><label>Blur</label><span class="val" id="blV">1</span></div><input type="range" id="bl" min="0" max="5" value="1" oninput="$('blV').textContent=this.value"></div><div class="slider"><div class="slider-head"><label>Simplify</label><span class="val" id="siV">2</span></div><input type="range" id="si" min="1" max="10" value="2" oninput="$('siV').textContent=this.value"></div><div class="slider"><div class="slider-head"><label>Smooth</label><span class="val" id="smV">0</span></div><input type="range" id="sm" min="0" max="10" value="0" oninput="$('smV').textContent=this.value"></div><label class="check"><input type="checkbox" id="inv"> Invert Colors</label></div><button class="btn btn-secondary" id="btnConvert" onclick="convert()" disabled>Convert</button><button class="btn btn-primary" id="btnExport" onclick="exportSVG()" disabled>⬇ Download SVG</button><div class="status" id="status"></div><div class="footer">Made by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div></div><div class="preview-panel" id="panelPreview"><p style="color:#888">Upload an image</p></div><div class="panel" id="panelInfo"><div class="sec-title">Output Info</div><div class="stats"><div class="stat"><label>Paths</label><div class="v" id="stP">-</div></div><div class="stat"><label>Size</label><div class="v" id="stS">-</div></div></div></div></div>
<script>const $=id=>document.getElementById(id);let mode='outline',imgData=null,svgRes=null;const up=$('up');up.ondragover=e=>{e.preventDefault();up.style.borderColor='#00AE42'};up.ondragleave=()=>up.style.borderColor='#2a2a2a';up.ondrop=e=>{e.preventDefault();up.style.borderColor='#2a2a2a';if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])};$('file').onchange=e=>{if(e.target.files.length)handleFile(e.target.files[0])};function handleFile(f){if(!f.type.startsWith('image/')){msg('error','Please upload an image');return}const reader=new FileReader();reader.onload=e=>{imgData=e.target.result;up.innerHTML='<img src="'+imgData+'">';up.classList.add('has-img');$('btnConvert').disabled=false;$('panelPreview').innerHTML='<img src="'+imgData+'">';msg('success','Image loaded')};reader.readAsDataURL(f)}function setMode(m){mode=m;document.querySelectorAll('.mode').forEach(e=>e.classList.toggle('active',e.dataset.m===m))}async function convert(){if(!imgData)return;$('btnConvert').disabled=true;$('btnConvert').textContent='Converting...';try{const r=await fetch('/api/image/convert',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:imgData,mode,threshold:parseInt($('th').value),blur:parseInt($('bl').value),simplify:parseInt($('si').value),smooth:parseInt($('sm').value),invert:$('inv').checked,singleLine:mode==='centerline'})});const d=await r.json();if(d.error)throw new Error(d.error);svgRes=d.svg;$('panelPreview').innerHTML=svgRes;$('stP').textContent=d.paths;$('stS').textContent=d.width+'x'+d.height;$('btnExport').disabled=false;msg('success','Converted!')}catch(e){msg('error',e.message)}finally{$('btnConvert').disabled=false;$('btnConvert').textContent='Convert'}}function exportSVG(){if(!svgRes)return;const blob=new Blob([svgRes],{type:'image/svg+xml'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='vectorized.svg';a.click();msg('success','SVG downloaded!')}function msg(type,text){const el=$('status');el.className='status '+type;el.textContent=text;setTimeout(()=>el.className='status',3000)}</script>
</body></html>"""

MAP_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Map to STL | SebGE Tools</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>*{box-sizing:border-box;margin:0;padding:0}:root{--g:#00AE42;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}body{font-family:system-ui;background:var(--bg);color:var(--t);height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:280px 1fr 260px;height:100vh}.panel{background:var(--c);padding:12px;display:flex;flex-direction:column;gap:8px;overflow-y:auto}.back{color:var(--t2);text-decoration:none;font-size:10px}.logo{display:flex;align-items:center;gap:8px;padding-bottom:8px;border-bottom:1px solid var(--br)}.logo-icon{width:32px;height:32px;background:linear-gradient(135deg,var(--g),#009639);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px}.logo h1{font-size:13px}.search{position:relative}.search input{width:100%;padding:10px 10px 10px 32px;background:var(--c2);border:1px solid var(--br);border-radius:6px;color:var(--t);font-size:14px}.search input:focus{outline:none;border-color:var(--g)}.search svg{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--t2);width:14px;height:14px}.search-btn{position:absolute;right:4px;top:50%;transform:translateY(-50%);background:var(--g);border:none;color:#fff;padding:6px 12px;border-radius:4px;font-size:10px;cursor:pointer}.coords{display:flex;gap:6px;margin-top:4px}.coord{flex:1;background:var(--c2);border-radius:4px;padding:6px 8px;font-size:10px}.coord label{color:var(--t2);font-size:8px}.coord span{color:var(--g);font-family:monospace}.sec{background:var(--c2);border-radius:8px;padding:10px}.sec-title{font-size:9px;font-weight:700;text-transform:uppercase;color:var(--g);margin-bottom:8px}.modes{display:flex;gap:4px}.mode{flex:1;padding:10px 4px;background:var(--c);border:2px solid var(--br);border-radius:6px;cursor:pointer;text-align:center}.mode:hover,.mode.active{border-color:var(--g)}.mode .icon{font-size:20px}.mode .name{font-size:9px;font-weight:600}.slider{margin-bottom:6px}.slider-head{display:flex;justify-content:space-between}.slider label{font-size:10px;color:var(--t2)}.slider .val{font-size:10px;color:var(--g);font-family:monospace}.slider input{width:100%;height:6px;background:var(--c);border-radius:3px;-webkit-appearance:none}.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;background:var(--g);border-radius:50%;cursor:pointer}.btn{padding:12px;border:none;border-radius:6px;font-size:12px;font-weight:700;cursor:pointer;width:100%}.btn-primary{background:var(--g);color:#fff}.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}.btn:disabled{opacity:.4}.btns{display:grid;grid-template-columns:1fr 1fr;gap:6px}.status{padding:8px;border-radius:6px;font-size:10px;display:none;text-align:center}.status.error{display:block;background:rgba(239,68,68,.1);color:#ef4444}.status.success{display:block;background:rgba(0,174,66,.1);color:var(--g)}.map-container{position:relative;background:#111}#map{width:100%;height:100%}#preview3d{flex:1;background:var(--c2);border-radius:8px;min-height:150px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px}.stat{background:var(--c2);border-radius:5px;padding:8px;text-align:center}.stat label{font-size:8px;color:var(--t2)}.stat .val{font-size:12px;color:var(--g);font-family:monospace}.footer{margin-top:auto;text-align:center;font-size:9px;color:var(--t2);padding-top:8px;border-top:1px solid var(--br)}.footer a{color:var(--g)}.spinner{width:14px;height:14px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .6s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:900px){body{height:auto}.app{display:block}.panel{min-height:50vh}.map-container{height:50vh}}</style></head><body>
<div class="app"><div class="panel" id="panelSettings"><a href="/" class="back">← Back</a><div class="logo"><div class="logo-icon">🗺️</div><div><h1>Map → STL</h1><small>3D PRINT YOUR WORLD</small></div></div><div class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><input type="text" id="loc" placeholder="Search location..." value="Dusseldorf" onkeypress="if(event.key==='Enter')search()"><button class="search-btn" onclick="search()">Go</button></div><div class="coords"><div class="coord"><label>LAT</label><br><span id="latV">51.2277</span></div><div class="coord"><label>LON</label><br><span id="lonV">6.7735</span></div></div><div class="sec"><div class="sec-title">Mode</div><div class="modes"><div class="mode active" data-mode="streets" onclick="setMode('streets')"><div class="icon">🛣️</div><div class="name">Streets</div></div><div class="mode" data-mode="city" onclick="setMode('city')"><div class="icon">🏙️</div><div class="name">City</div></div><div class="mode" data-mode="terrain" onclick="setMode('terrain')"><div class="icon">🏔️</div><div class="name">Terrain</div></div></div></div><div class="sec"><div class="sec-title">Settings</div><div class="slider"><div class="slider-head"><label>Radius</label><span class="val" id="radV">500m</span></div><input type="range" id="rad" min="100" max="3000" value="500" step="50" oninput="updRad()"></div><div class="slider"><div class="slider-head"><label>Model Size</label><span class="val" id="sizeV">80mm</span></div><input type="range" id="size" min="30" max="200" value="80" step="5" oninput="$('sizeV').textContent=this.value+'mm'"></div><div class="slider"><div class="slider-head"><label>Height</label><span class="val" id="htV">2mm</span></div><input type="range" id="ht" min="1" max="10" value="2" step="0.5" oninput="$('htV').textContent=this.value+'mm'"></div></div><div class="status" id="status"></div></div><div class="map-container" id="panelMap"><div id="map"></div></div><div class="panel" id="panelPreview"><div class="sec-title">3D Preview</div><div id="preview3d">Click Preview</div><div class="stats"><div class="stat"><label>Mode</label><div class="val" id="statMode">Streets</div></div><div class="stat"><label>Data</label><div class="val" id="statData">-</div></div></div><div class="btns"><button class="btn btn-secondary" id="btnPreview" onclick="preview()">Preview</button><button class="btn btn-primary" id="btnExport" onclick="exportSTL()">⬇ Export STL</button></div><div class="footer">Made by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div></div></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>const $=id=>document.getElementById(id);let map,mapMarker,circle,lat=51.2277,lon=6.7735,mode='streets';let scene,camera,renderer,mesh;function init(){map=L.map('map',{zoomControl:false}).setView([lat,lon],14);L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{maxZoom:19}).addTo(map);L.control.zoom({position:'topright'}).addTo(map);updateMap();map.on('click',e=>{lat=e.latlng.lat;lon=e.latlng.lng;updateMap();updateCoords()});init3D();setTimeout(()=>map.invalidateSize(),100)}function init3D(){const c=$('preview3d');scene=new THREE.Scene();scene.background=new THREE.Color(0x1e1e1e);camera=new THREE.PerspectiveCamera(45,c.clientWidth/Math.max(c.clientHeight,150),0.1,1000);camera.position.set(80,80,80);renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(c.clientWidth,Math.max(c.clientHeight,150));scene.add(new THREE.AmbientLight(0xffffff,0.6));const dir=new THREE.DirectionalLight(0xffffff,0.8);dir.position.set(50,100,50);scene.add(dir)}function updateMap(){const r=+$('rad').value;if(mapMarker)map.removeLayer(mapMarker);if(circle)map.removeLayer(circle);mapMarker=L.marker([lat,lon]).addTo(map);circle=L.circle([lat,lon],{radius:r,color:'#00AE42',fillOpacity:.08,weight:2}).addTo(map)}function updRad(){const r=$('rad').value;$('radV').textContent=r>=1000?(r/1000)+'km':r+'m';if(circle)circle.setRadius(+r)}function updateCoords(){$('latV').textContent=lat.toFixed(5);$('lonV').textContent=lon.toFixed(5)}function setMode(m){mode=m;document.querySelectorAll('.mode').forEach(el=>el.classList.toggle('active',el.dataset.mode===m));$('statMode').textContent=m}async function search(){const q=$('loc').value.trim();if(!q)return;try{const r=await fetch('/api/geocode?q='+encodeURIComponent(q));const d=await r.json();if(d.error)throw new Error(d.error);lat=d.lat;lon=d.lon;map.setView([lat,lon],14);updateMap();updateCoords();msg('success','Found: '+d.name.substring(0,25))}catch(e){msg('error',e.message)}}async function preview(){const btn=$('btnPreview');btn.disabled=true;btn.innerHTML='<div class="spinner"></div>';try{const r=await fetch('/api/map/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon,radius:+$('rad').value,size:+$('size').value,height:+$('ht').value,lineWidth:1.2,mode})});const d=await r.json();if(d.error)throw new Error(d.error);load3DPreview(d.vertices,d.faces);$('statData').textContent=d.count;msg('success','Preview loaded!')}catch(e){msg('error',e.message)}finally{btn.disabled=false;btn.innerHTML='Preview'}}function load3DPreview(verts,faces){const c=$('preview3d');if(!renderer||renderer.domElement.width!==c.clientWidth){scene=new THREE.Scene();scene.background=new THREE.Color(0x1e1e1e);camera=new THREE.PerspectiveCamera(45,c.clientWidth/c.clientHeight,0.1,1000);camera.position.set(80,80,80);renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(c.clientWidth,c.clientHeight);scene.add(new THREE.AmbientLight(0xffffff,0.6));const dir=new THREE.DirectionalLight(0xffffff,0.8);dir.position.set(50,100,50);scene.add(dir)}if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose()}const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));geom.setIndex(faces);geom.computeVertexNormals();const mat=new THREE.MeshPhongMaterial({color:0x00AE42,flatShading:true});mesh=new THREE.Mesh(geom,mat);geom.computeBoundingBox();const center=new THREE.Vector3();geom.boundingBox.getCenter(center);mesh.position.sub(center);const maxDim=Math.max(geom.boundingBox.max.x-geom.boundingBox.min.x,geom.boundingBox.max.y-geom.boundingBox.min.y);mesh.scale.set(60/maxDim,60/maxDim,60/maxDim);scene.add(mesh);c.innerHTML='';c.appendChild(renderer.domElement);let angle=0;function animate(){requestAnimationFrame(animate);angle+=0.005;camera.position.x=Math.sin(angle)*100;camera.position.z=Math.cos(angle)*100;camera.lookAt(0,0,0);renderer.render(scene,camera)}animate()}async function exportSTL(){const btn=$('btnExport');btn.disabled=true;btn.innerHTML='<div class="spinner"></div>';try{const r=await fetch('/api/map/stl',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon,radius:+$('rad').value,size:+$('size').value,height:+$('ht').value,lineWidth:1.2,mode})});if(!r.ok){const err=await r.json();throw new Error(err.error)}const blob=await r.blob();const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='map.stl';a.click();msg('success','STL downloaded!')}catch(e){msg('error',e.message)}finally{btn.disabled=false;btn.innerHTML='⬇ Export STL'}}function msg(type,text){const el=$('status');el.className='status '+type;el.textContent=text;setTimeout(()=>el.className='status',4000)}document.addEventListener('DOMContentLoaded',init);</script>
</body></html>'''


PRINT_HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HueForge Print | SebGE Tools</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}:root{--g:#00AE42;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}body{font-family:system-ui;background:var(--bg);color:var(--t);min-height:100vh}.app{display:grid;grid-template-columns:300px 1fr 280px;height:100vh}.panel{background:var(--c);padding:14px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}.back{color:var(--t2);text-decoration:none;font-size:10px}.back:hover{color:var(--g)}.logo{display:flex;align-items:center;gap:10px;padding-bottom:10px;border-bottom:1px solid var(--br)}.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--g),#009639);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px}.logo h1{font-size:14px}.logo small{display:block;font-size:8px;color:var(--g)}.sec{background:var(--c2);border-radius:8px;padding:12px}.sec-title{font-size:9px;color:var(--g);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;font-weight:600}.upload{border:2px dashed var(--br);border-radius:8px;padding:24px 12px;text-align:center;cursor:pointer}.upload:hover{border-color:var(--g)}.upload p{font-size:11px;color:var(--t2)}.upload.has-img{padding:10px}.upload img{max-width:100%;max-height:120px;border-radius:6px}#file{display:none}.color-btns{display:flex;gap:6px}.color-btn{flex:1;padding:10px;background:var(--c);border:2px solid var(--br);border-radius:6px;cursor:pointer;text-align:center;font-size:13px;font-weight:600;color:var(--t)}.color-btn:hover,.color-btn.active{border-color:var(--g);background:rgba(0,174,66,.1)}.layer-row{display:flex;align-items:center;gap:10px;padding:10px;background:var(--c);border-radius:8px;margin-bottom:8px}.layer-color{width:40px;height:40px;border-radius:8px;border:2px solid var(--br);cursor:pointer;padding:0}.layer-info{flex:1}.layer-name{font-size:11px;font-weight:600;margin-bottom:4px}.layer-thresh{width:100%;height:5px;background:var(--c2);border-radius:3px;-webkit-appearance:none}.layer-thresh::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--g);border-radius:50%;cursor:pointer}.slider{margin-bottom:8px}.slider-head{display:flex;justify-content:space-between;margin-bottom:4px}.slider label{font-size:10px;color:var(--t2)}.slider .val{font-size:10px;color:var(--g);font-family:monospace}.slider input{width:100%;height:5px;background:var(--c);border-radius:3px;-webkit-appearance:none}.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;background:var(--g);border-radius:50%;cursor:pointer}.btn{padding:12px;border:none;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;width:100%}.btn-primary{background:var(--g);color:#fff}.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}.btn:disabled{opacity:.4;cursor:not-allowed}.btn-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.preview-panel{background:#222;display:flex;flex-direction:column}.preview-header{padding:12px;border-bottom:1px solid var(--br)}.preview-header h2{font-size:13px}.preview-area{flex:1;display:flex;align-items:center;justify-content:center;padding:20px;background:#181818}#canvas{max-width:100%;max-height:100%;border-radius:8px}.instr{background:var(--c);font-family:monospace;font-size:10px;padding:12px;border-radius:8px;white-space:pre-wrap;line-height:1.6;max-height:180px;overflow-y:auto}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px}.stat{background:var(--c2);border-radius:6px;padding:8px;text-align:center}.stat label{font-size:8px;color:var(--t2);text-transform:uppercase}.stat .v{font-size:12px;color:var(--g);font-family:monospace;font-weight:600}.status{padding:8px;border-radius:6px;font-size:10px;display:none;text-align:center}.status.error{display:block;background:rgba(239,68,68,.1);color:#ef4444}.status.success{display:block;background:rgba(0,174,66,.1);color:var(--g)}.footer{margin-top:auto;text-align:center;font-size:9px;color:var(--t2);padding-top:10px;border-top:1px solid var(--br)}.footer a{color:var(--g)}.spinner{width:14px;height:14px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .6s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:950px){.app{display:block;height:auto}.panel,.preview-panel{min-height:50vh}}
</style></head><body>
<div class="app">
<div class="panel"><a href="/" class="back">← Back</a><div class="logo"><div class="logo-icon">🖼️</div><div><h1>HueForge Print</h1><small>LITHOPHANE STL</small></div></div>
<div class="upload" id="upload" onclick="document.getElementById('file').click()"><p>📤 Drop image or click</p></div><input type="file" id="file" accept="image/*">
<div class="sec"><div class="sec-title">Colors</div><div class="color-btns"><div class="color-btn active" onclick="setColors(2)">2</div><div class="color-btn" onclick="setColors(3)">3</div><div class="color-btn" onclick="setColors(4)">4</div></div></div>
<div class="sec"><div class="sec-title">Thresholds</div><div id="layers"></div></div>
<div class="sec"><div class="sec-title">Model</div>
<div class="slider"><div class="slider-head"><label>Width</label><span class="val" id="wV">100mm</span></div><input type="range" id="w" min="50" max="200" value="100" step="5" oninput="$('wV').textContent=this.value+'mm'"></div>
<div class="slider"><div class="slider-head"><label>Total Height</label><span class="val" id="hV">2.4mm</span></div><input type="range" id="h" min="1.6" max="4" value="2.4" step="0.2" oninput="$('hV').textContent=this.value+'mm';updateInstr()"></div>
<div class="slider"><div class="slider-head"><label>Base</label><span class="val" id="bV">0.6mm</span></div><input type="range" id="b" min="0.4" max="1.2" value="0.6" step="0.1" oninput="$('bV').textContent=this.value+'mm';updateInstr()"></div>
</div>
<div class="status" id="status"></div></div>
<div class="preview-panel"><div class="preview-header"><h2>Preview</h2></div><div class="preview-area"><canvas id="canvas"></canvas><div id="ph" style="color:#666">Upload image</div></div></div>
<div class="panel"><div class="sec-title">Instructions</div><div class="instr" id="instr">1. Upload image
2. Choose colors
3. Adjust thresholds
4. Export STL</div>
<div class="stats"><div class="stat"><label>W</label><div class="v" id="stW">-</div></div><div class="stat"><label>H</label><div class="v" id="stH">-</div></div><div class="stat"><label>L</label><div class="v" id="stL">-</div></div></div>
<button class="btn btn-primary" id="btnSTL" onclick="exportSTL()" disabled>⬇ Download STL</button>
<div class="footer">By <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div></div></div>
<script>
const $=id=>document.getElementById(id);let imgData=null,grayData=null,numColors=2;
// Default: evenly distributed thresholds (pixel values where color changes)
const defaultLayers={
    2:[{c:'#FFFFFF',t:128,n:'Light'},{c:'#222222',t:0,n:'Dark'}],
    3:[{c:'#FFFFFF',t:170,n:'Light'},{c:'#888888',t:85,n:'Medium'},{c:'#222222',t:0,n:'Dark'}],
    4:[{c:'#FFFFFF',t:192,n:'Lightest'},{c:'#BBBBBB',t:128,n:'Light'},{c:'#666666',t:64,n:'Dark'},{c:'#222222',t:0,n:'Darkest'}]
};
let layers=JSON.parse(JSON.stringify(defaultLayers[2]));

function renderLayers(){
    const el=$('layers');el.innerHTML='';
    layers.forEach((l,i)=>{
        const isLast=i===layers.length-1;
        el.innerHTML+=`<div class="layer-row">
            <input type="color" class="layer-color" value="${l.c}" onchange="layers[${i}].c=this.value;updatePreview()">
            <div class="layer-info">
                <div class="layer-name">${l.n||'Color '+(i+1)}</div>
                ${isLast?'<div style="font-size:9px;color:#666">Darkest pixels (base)</div>'
                :`<input type="range" class="layer-thresh" min="1" max="254" value="${l.t}" 
                    oninput="layers[${i}].t=+this.value;updatePreview();updateSliderLabel(${i})">
                <div class="slider-label" id="sl${i}" style="font-size:9px;color:#666;margin-top:2px">Pixels ≥ ${l.t}</div>`}
            </div>
        </div>`;
    });
}

function updateSliderLabel(i){
    const label=$('sl'+i);
    if(label)label.textContent='Pixels ≥ '+layers[i].t;
}

function setColors(n){
    numColors=n;
    document.querySelectorAll('.color-btn').forEach((b,i)=>b.classList.toggle('active',i===n-2));
    layers=JSON.parse(JSON.stringify(defaultLayers[n]));
    renderLayers();
    updatePreview();
    updateInstr();
}

const up=$('upload');up.ondragover=e=>{e.preventDefault();up.style.borderColor='#00AE42'};up.ondragleave=()=>up.style.borderColor='#2a2a2a';up.ondrop=e=>{e.preventDefault();up.style.borderColor='#2a2a2a';if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])};
$('file').onchange=e=>{if(e.target.files.length)handleFile(e.target.files[0])};
function handleFile(f){if(!f.type.startsWith('image/'))return;const r=new FileReader();r.onload=e=>{imgData=e.target.result;up.innerHTML='<img src="'+imgData+'">';up.classList.add('has-img');loadPreview(imgData)};r.readAsDataURL(f)}
function loadPreview(src){const img=new Image();img.onload=()=>{const c=$('canvas'),x=c.getContext('2d'),maxS=500;let w=img.width,h=img.height;if(w>maxS||h>maxS){const s=maxS/Math.max(w,h);w=Math.round(w*s);h=Math.round(h*s)}c.width=w;c.height=h;x.drawImage(img,0,0,w,h);const d=x.getImageData(0,0,w,h);grayData=new Uint8Array(w*h);for(let i=0;i<w*h;i++)grayData[i]=Math.round(0.299*d.data[i*4]+0.587*d.data[i*4+1]+0.114*d.data[i*4+2]);c.style.display='block';$('ph').style.display='none';$('btnSTL').disabled=false;updatePreview()};img.src=src}
function hexToRgb(h){return[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]}

function updatePreview(){
    if(!grayData)return;
    const c=$('canvas'),x=c.getContext('2d'),w=c.width,h=c.height,d=x.createImageData(w,h);
    
    // Sort by threshold descending (highest threshold first = brightest color)
    const sorted=[...layers].sort((a,b)=>(b.t||0)-(a.t||0));
    
    for(let i=0;i<w*h;i++){
        const g=grayData[i];
        // Find which color: check from brightest to darkest
        let rgb=hexToRgb(sorted[sorted.length-1].c); // Default darkest
        for(const l of sorted){
            if(g>=l.t){
                rgb=hexToRgb(l.c);
                break;
            }
        }
        d.data[i*4]=rgb[0];d.data[i*4+1]=rgb[1];d.data[i*4+2]=rgb[2];d.data[i*4+3]=255;
    }
    x.putImageData(d,0,0);
}

function updateInstr(){
    const t=+$('h').value,b=+$('b').value,layerH=0.12;
    const totalLayers=Math.round(t/layerH);
    const reliefH=t-b;
    
    $('stL').textContent=totalLayers;
    
    // Sort by threshold descending for display
    const sorted=[...layers].map((l,i)=>({...l,i})).sort((a,b)=>(b.t||0)-(a.t||0));
    
    let txt=`Total: ${t}mm | ${totalLayers} layers\n\n`;
    sorted.forEach((l,i)=>{
        // Higher threshold = brighter pixels = lower Z (printed first)
        const heightFrac=1-(l.t/255);
        const z=b+(reliefH*heightFrac);
        const layerNum=Math.round(z/layerH);
        txt+=`${i+1}. ${l.c} - Layer ${layerNum} (Z≈${z.toFixed(1)}mm)\n`;
    });
    txt+=`\nBrightest first, darkest last.`;
    $('instr').textContent=txt;
}

async function exportSTL(){if(!imgData)return;$('btnSTL').disabled=true;$('btnSTL').innerHTML='<span class="spinner"></span>';try{const r=await fetch('/api/print/stl',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:imgData,width:+$('w').value,total_height:+$('h').value,base_height:+$('b').value,border:2,contrast:1.5,denoise:2})});if(!r.ok)throw new Error('Failed');const blob=await r.blob(),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='hueforge.stl';a.click();$('stW').textContent=$('w').value+'mm';$('stH').textContent=Math.round(+$('w').value*$('canvas').height/$('canvas').width)+'mm'}catch(e){alert(e.message)}finally{$('btnSTL').disabled=false;$('btnSTL').innerHTML='⬇ Download STL'}}
renderLayers();updateInstr();
</script></body></html>"""
# === SHADOW (Silhouette -> STL for Shadow Box) ===
def _shadow_extract_geometry(img, height_mm, threshold, smooth, foot_x, foot_y, foot_h, foot_width=57.969, cleanup=0.0, max_points=3000):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)
    alpha, rgb = arr[:, :, 3], arr[:, :, :3]
    gray = np.mean(rgb, axis=2)
    mask = (alpha > 128) & (gray < threshold) if alpha.min() < 250 else gray < threshold
    if not mask.any():
        raise ValueError("No silhouette found")
    if cleanup and cleanup > 0:
        from scipy.ndimage import binary_closing
        it = int(min(3, max(1, round(cleanup))))
        mask = binary_closing(mask, iterations=it)
    rows, cols = np.where(mask)
    img_height = rows.max() - rows.min()
    img_width = cols.max() - cols.min()
    if img_height <= 0:
        raise ValueError("Invalid silhouette height")
    scale = float(height_mm) / float(img_height)
    y_max = rows.max()
    x_center = (cols.min() + cols.max()) / 2.0
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        raise ValueError("No contours found")
    min_area_px = float(img_height * img_width) * 0.0002
    all_polygons = []
    for contour in contours:
        if len(contour) < 20:
            continue
        n = len(contour)
        area = 0.5 * abs(sum(contour[i, 0] * contour[(i + 1) % n, 1] - contour[(i + 1) % n, 0] * contour[i, 1] for i in range(n)))
        if area < min_area_px:
            continue
        from scipy.ndimage import uniform_filter1d
        w = max(3, int(len(contour) * 0.0015 * max(0.0, smooth)))
        if w > 3:
            sy = uniform_filter1d(contour[:, 0], size=w, mode="wrap")
            sx = uniform_filter1d(contour[:, 1], size=w, mode="wrap")
        else:
            sy, sx = contour[:, 0], contour[:, 1]
        if len(sy) > max_points:
            step = max(1, len(sy) // max_points)
            sy = sy[::step]
            sx = sx[::step]
        pts = [(float((x - x_center) * scale), float((y_max - y) * scale)) for y, x in zip(sy, sx)]
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0.5:
            continue
        tol = float(max(0.0, (smooth - 0.5) * 0.05))
        if tol > 0:
            poly = poly.simplify(tol, preserve_topology=True)
            if not poly.is_valid:
                poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0.5:
            all_polygons.append(poly)
    if not all_polygons:
        raise ValueError("No valid shapes")
    all_polygons.sort(key=lambda p: p.area, reverse=True)
    outer = all_polygons[0]
    holes = []
    separate = []
    for poly in all_polygons[1:]:
        try:
            if outer.contains(poly.representative_point()):
                holes.append(poly)
            else:
                separate.append(poly)
        except Exception:
            separate.append(poly)
    sil = outer
    for h in holes:
        try:
            sil = sil.difference(h)
        except Exception:
            pass
    sil_combined = unary_union([sil] + separate) if separate else sil
    if not sil_combined.is_valid:
        sil_combined = sil_combined.buffer(0)
    combined = sil_combined
    foot_poly = None
    if foot_h and foot_h > 0:
        bounds = sil_combined.bounds
        sil_bottom = bounds[1]
        sil_center_x = (bounds[0] + bounds[2]) / 2.0
        foot_bottom = sil_bottom - float(foot_h) + float(foot_y)
        foot_top = sil_bottom + float(foot_y) + 0.5
        foot_center = sil_center_x + float(foot_x)
        foot_poly = box(foot_center - foot_width / 2.0, foot_bottom, foot_center + foot_width / 2.0, foot_top)
        combined = unary_union([sil_combined, foot_poly])
        if not combined.is_valid:
            combined = combined.buffer(0)
    def _rings(g):
        if g is None or getattr(g, "is_empty", True):
            return []
        if g.geom_type == "Polygon":
            return [[list(map(float, xy)) for xy in g.exterior.coords]]
        if g.geom_type == "MultiPolygon":
            return [[[float(x), float(y)] for x, y in p.exterior.coords] for p in g.geoms]
        return []
    preview_rings = _rings(combined)
    foot_ring = _rings(foot_poly)[0] if foot_poly is not None and not foot_poly.is_empty else None
    return combined, preview_rings, foot_ring, float(img_width * scale)

SHADOW_HTML='''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Family Shadow Generator | SebGE Tools</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}:root{--g:#00AE42;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}body{font-family:system-ui;background:var(--bg);color:var(--t);min-height:100vh}.app{display:grid;grid-template-columns:340px 1fr;height:100vh}.panel{background:var(--c);padding:14px;overflow-y:auto;display:flex;flex-direction:column;gap:12px}.back{color:var(--t2);text-decoration:none;font-size:10px}.logo{display:flex;align-items:center;gap:10px;padding-bottom:10px;border-bottom:1px solid var(--br)}.logo-icon{width:40px;height:40px;background:linear-gradient(135deg,var(--g),#009639);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px}.logo h1{font-size:15px}.logo small{display:block;font-size:8px;color:var(--g)}.step{background:var(--c2);border-radius:10px;padding:14px}.step-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}.step-num{width:24px;height:24px;background:var(--g);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}.step-title{font-size:12px;font-weight:600}.step p{font-size:10px;color:var(--t2);line-height:1.5;margin-bottom:10px}.gpt-btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;padding:12px;background:linear-gradient(135deg,#10a37f,#1a7f64);color:#fff;text-decoration:none;border-radius:8px;font-size:12px;font-weight:600}.upload{border:2px dashed var(--br);border-radius:10px;padding:20px 15px;text-align:center;cursor:pointer}.upload:hover{border-color:var(--g)}.upload.has-image{padding:8px;border-style:solid;border-color:var(--g)}.upload img{max-width:100%;max-height:100px;border-radius:6px}.upload p{font-size:11px;color:var(--t2)}.upload .icon{font-size:28px;margin-bottom:6px}#file{display:none}.settings{display:none}.settings.visible{display:block}.slider{margin-bottom:6px}.slider-head{display:flex;justify-content:space-between;margin-bottom:3px}.slider label{font-size:10px;color:var(--t2)}.slider .val{font-size:10px;color:var(--g);font-family:monospace}.slider input{width:100%;height:5px;background:var(--c);border-radius:3px;-webkit-appearance:none}.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--g);border-radius:50%;cursor:pointer}.btn{padding:12px;border:none;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;width:100%}.btn-primary{background:var(--g);color:#fff}.btn:disabled{opacity:.4}.preview-panel{background:#f0f0f0;display:flex;align-items:center;justify-content:center;padding:20px}.preview-container{position:relative;width:100%;height:100%}#preview3d{width:100%;height:100%;cursor:grab}.stats{display:flex;gap:8px;margin-bottom:8px}.stat{background:var(--c2);border-radius:6px;padding:8px 12px;text-align:center;flex:1}.stat label{font-size:8px;color:var(--t2)}.stat .val{font-size:13px;color:var(--g);font-family:monospace}.status{padding:10px;border-radius:6px;font-size:11px;display:none;text-align:center;margin-top:8px}.status.error{display:block;background:rgba(239,68,68,.1);color:#ef4444}.status.success{display:block;background:rgba(0,174,66,.1);color:var(--g)}.footer{margin-top:auto;text-align:center;font-size:9px;color:var(--t2);padding-top:10px;border-top:1px solid var(--br)}.footer a{color:var(--g)}.divider{border:none;border-top:1px solid var(--br);margin:4px 0}.section-label{font-size:9px;color:var(--g);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;font-weight:600}.foot-info{font-size:9px;color:var(--t2);margin-bottom:8px}.spinner{width:14px;height:14px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .6s linear infinite;display:inline-block}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:900px){.app{display:block}.panel,.preview-panel{min-height:50vh}}
</style></head><body>
<div class="app">
<div class="panel" id="panelSettings">
<a href="/" class="back">← Back to Tools</a>
<div class="logo"><div class="logo-icon">👨‍👩‍👧</div><div><h1>Family Shadow</h1><small>SILHOUETTE TO STL</small></div></div>
<div class="step" style="background:rgba(0,174,66,0.1);border:1px solid rgba(0,174,66,0.3)"><p style="margin:0;font-size:11px">🖼️ <strong>Need a Shadow Box frame?</strong> <a href="https://makerworld.com/de/models/2405762" target="_blank" style="color:var(--g)">Download my Customizable Shadow Box on MakerWorld</a></p></div>
<div class="step"><div class="step-header"><div class="step-num">1</div><div class="step-title">Create Your Silhouette</div></div><p>Use our AI to generate a family silhouette.</p><a href="https://chatgpt.com/g/g-699077b0e53c8191a6cfbb033250c030-family-silhouette-cutter" target="_blank" class="gpt-btn">Open Silhouette Creator</a></div>
<div class="step"><div class="step-header"><div class="step-num">2</div><div class="step-title">Upload Image</div></div><div class="upload" id="upload" onclick="document.getElementById('file').click()"><div class="icon">📤</div><p>Drop PNG here or click</p></div><input type="file" id="file" accept="image/png,image/jpeg,image/webp"></div>
<div class="step settings" id="settingsPanel">
<div class="step-header"><div class="step-num">3</div><div class="step-title">Settings</div></div>
<div class="section-label">Silhouette</div>
<div class="slider"><div class="slider-head"><label>Height</label><span class="val" id="heightV">140mm</span></div><input type="range" id="height" min="80" max="170" value="140" oninput="updateSettings()"></div>
<div class="slider"><div class="slider-head"><label>Thickness</label><span class="val" id="thicknessV">2mm</span></div><input type="range" id="thickness" min="1.5" max="4" value="2" step="0.5" oninput="updateSettings()"></div>
<hr class="divider">
<div class="section-label">Mounting Foot (58mm fixed width)</div>
<div class="foot-info">Adjust position to connect all parts</div>
<div class="slider"><div class="slider-head"><label>↕ Vertical</label><span class="val" id="footYV">0mm</span></div><input type="range" id="footY" min="0" max="40" value="0" oninput="updateSettings()"></div>
<div class="slider"><div class="slider-head"><label>↔ Horizontal</label><span class="val" id="footXV">0mm</span></div><input type="range" id="footX" min="-40" max="40" value="0" oninput="updateSettings()"></div>
<div class="slider"><div class="slider-head"><label>Foot Height</label><span class="val" id="footHV">4mm</span></div><input type="range" id="footH" min="0" max="20" value="4" oninput="updateSettings()"></div>
<hr class="divider">
<div class="section-label">Image Processing</div>
<div class="slider"><div class="slider-head"><label>Smoothness</label><span class="val" id="smoothV">0.8</span></div><input type="range" id="smooth" min="0.0" max="3" value="0.8" step="0.1" oninput="updateSettings()"></div>
<div class="slider"><div class="slider-head"><label>Cleanup</label><span class="val" id="cleanupV">0</span></div><input type="range" id="cleanup" min="0" max="3" value="0" step="1" oninput="updateSettings()"></div>
<div class="slider"><div class="slider-head"><label>Detail (Points)</label><span class="val" id="maxPointsV">3000</span></div><input type="range" id="maxPoints" min="800" max="8000" value="3000" step="100" oninput="updateSettings()"></div>
<div class="slider"><div class="slider-head"><label>Threshold</label><span class="val" id="thresholdV">128</span></div><input type="range" id="threshold" min="10" max="245" value="128" oninput="updateSettings()"></div>
</div>
<div class="stats"><div class="stat"><label>Width</label><div class="val" id="statWidth">-</div></div><div class="stat"><label>Height</label><div class="val" id="statHeight">-</div></div></div>
<button class="btn btn-primary" onclick="exportSTL()" id="btnExport" disabled>⬇ Download STL</button>
<div class="status" id="status"></div>
<div class="footer">Made by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div>
</div>
<div class="preview-panel" id="panelPreview"><div class="preview-container"><canvas id="preview3d"></canvas></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
let imageData=null,scene,camera,renderer,mesh,processTimeout=null;
let orbit={theta:0.8,phi:1.15,radius:260,target:new THREE.Vector3(0,70,0),drag:false,lastX:0,lastY:0};
const upload=$('upload');
upload.ondragover=e=>{e.preventDefault();upload.style.borderColor='#00AE42'};
upload.ondragleave=()=>upload.style.borderColor='#2a2a2a';
upload.ondrop=e=>{e.preventDefault();upload.style.borderColor='#2a2a2a';if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])};
$('file').onchange=e=>{if(e.target.files.length)handleFile(e.target.files[0])};
function handleFile(file){if(!file.type.startsWith('image/')){msg('error','Please upload an image');return}const reader=new FileReader();reader.onload=e=>{imageData=e.target.result;upload.innerHTML='<img src="'+imageData+'">';upload.classList.add('has-image');$('settingsPanel').classList.add('visible');processImage()};reader.readAsDataURL(file)}
function updateSettings(){$('heightV').textContent=$('height').value+'mm';$('thicknessV').textContent=$('thickness').value+'mm';$('footYV').textContent=$('footY').value+'mm up';$('footXV').textContent=($('footX').value>=0?'+':'')+$('footX').value+'mm';$('footHV').textContent=$('footH').value+'mm';$('smoothV').textContent=parseFloat($('smooth').value).toFixed(1);$('cleanupV').textContent=$('cleanup').value;$('maxPointsV').textContent=$('maxPoints').value;$('thresholdV').textContent=$('threshold').value;if(processTimeout)clearTimeout(processTimeout);if(imageData)processTimeout=setTimeout(processImage,400)}
async function processImage(){if(!imageData)return;msg('success','Processing...');try{const r=await fetch('/api/shadow/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:imageData,height:parseFloat($('height').value),thickness:parseFloat($('thickness').value),threshold:parseInt($('threshold').value),smooth:parseFloat($('smooth').value),footX:parseFloat($('footX').value),footY:parseFloat($('footY').value),footH:parseFloat($('footH').value),cleanup:parseFloat($('cleanup').value),maxPoints:parseInt($('maxPoints').value)})});const d=await r.json();if(d.error)throw new Error(d.error);$('statWidth').textContent=d.width.toFixed(0)+'mm';$('statHeight').textContent=d.height.toFixed(0)+'mm';$('btnExport').disabled=false;if(d.vertices&&d.faces){init3D();loadPreview(d.vertices,d.faces)}msg('success','Ready!')}catch(e){msg('error',e.message)}}
function clamp(v,min,max){return Math.max(min,Math.min(max,v))}
function updateCamera(){const t=orbit.target;camera.position.x=t.x+orbit.radius*Math.sin(orbit.phi)*Math.sin(orbit.theta);camera.position.z=t.z+orbit.radius*Math.sin(orbit.phi)*Math.cos(orbit.theta);camera.position.y=t.y+orbit.radius*Math.cos(orbit.phi);camera.lookAt(t)}
function init3D(){if(renderer)return;const container=$('preview3d');scene=new THREE.Scene();scene.background=new THREE.Color(0xf0f0f0);camera=new THREE.PerspectiveCamera(45,1,0.1,1000);const wrap=$('panelPreview');camera.aspect=wrap.clientWidth/wrap.clientHeight;camera.updateProjectionMatrix();renderer=new THREE.WebGLRenderer({canvas:container,antialias:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.setSize(wrap.clientWidth,wrap.clientHeight);scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1.2));const dir=new THREE.DirectionalLight(0xffffff,0.9);dir.position.set(80,140,90);scene.add(dir);setupDragOrbit();animate()}
function setupDragOrbit(){const c=$('preview3d');updateCamera();c.addEventListener('pointerdown',e=>{orbit.drag=true;orbit.lastX=e.clientX;orbit.lastY=e.clientY;c.setPointerCapture(e.pointerId);c.style.cursor='grabbing'});c.addEventListener('pointermove',e=>{if(!orbit.drag)return;const dx=e.clientX-orbit.lastX,dy=e.clientY-orbit.lastY;orbit.lastX=e.clientX;orbit.lastY=e.clientY;orbit.theta-=dx*0.006;orbit.phi=clamp(orbit.phi-dy*0.006,0.25,Math.PI-0.25);updateCamera()});c.addEventListener('pointerup',()=>{orbit.drag=false;$('preview3d').style.cursor='grab'});c.addEventListener('pointercancel',()=>{orbit.drag=false});c.addEventListener('wheel',e=>{e.preventDefault();orbit.radius=clamp(orbit.radius+e.deltaY*0.2,120,600);updateCamera()},{passive:false})}
function animate(){requestAnimationFrame(animate);if(renderer)renderer.render(scene,camera)}
function loadPreview(verts,faces){if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose()}const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));geom.setIndex(faces);geom.computeVertexNormals();mesh=new THREE.Mesh(geom,new THREE.MeshPhongMaterial({color:0x1a1a1a,flatShading:false,shininess:30}));geom.computeBoundingBox();const center=new THREE.Vector3();geom.boundingBox.getCenter(center);mesh.position.set(-center.x,0,-center.z);scene.add(mesh)}
async function exportSTL(){if(!imageData)return;$('btnExport').disabled=true;$('btnExport').innerHTML='<span class="spinner"></span> Exporting...';try{const r=await fetch('/api/shadow/stl',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:imageData,height:parseFloat($('height').value),thickness:parseFloat($('thickness').value),threshold:parseInt($('threshold').value),smooth:parseFloat($('smooth').value),footX:parseFloat($('footX').value),footY:parseFloat($('footY').value),footH:parseFloat($('footH').value),cleanup:parseFloat($('cleanup').value),maxPoints:parseInt($('maxPoints').value)})});if(!r.ok)throw new Error('Export failed');const blob=await r.blob(),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='family_silhouette.stl';a.click();msg('success','Downloaded!')}catch(e){msg('error',e.message)}finally{$('btnExport').disabled=false;$('btnExport').innerHTML='⬇ Download STL'}}
function msg(type,text){const el=$('status');el.className='status '+type;el.textContent=text;if(type==='success')setTimeout(()=>el.className='status',2000)}
window.addEventListener('resize',()=>{if(renderer){const c=$('panelPreview');camera.aspect=c.clientWidth/c.clientHeight;camera.updateProjectionMatrix();renderer.setSize(c.clientWidth,c.clientHeight);updateCamera()}});
</script>
</body></html>'''

# === ROUTES ===
@app.route('/')
def landing():
    return LANDING

@app.route('/map')
def map_tool():
    return MAP_HTML

@app.route('/image')
def image_tool():
    return IMAGE_HTML

@app.route('/print')
def print_tool():
    return PRINT_HTML


@app.route('/shadow')
def shadow_tool():
    return SHADOW_HTML

@app.route('/api/geocode')
def api_geo():
    q=request.args.get('q','')
    r=geocode(q) if q else None
    return jsonify(r) if r else (jsonify({'error':'Not found'}),404)

@app.route('/api/map/preview',methods=['POST'])
def api_map_preview():
    try:
        d=request.json;lat,lon,radius,size,height=d['lat'],d['lon'],d['radius'],d['size'],d['height']
        mode=d['mode'];lw=d.get('lineWidth',1.2)
        if mode=='streets':
            gdf=fetch_streets(lat,lon,radius);mesh=create_streets_mesh(gdf,lat,lon,radius,size,height,lw);count=len(gdf)
        elif mode=='city':
            mesh=create_city_mesh(lat,lon,radius,size,height,lw,1.0,1.5);count=len(fetch_buildings(lat,lon,radius))
        else:
            mesh=create_terrain_mesh(lat,lon,radius,size,height*5,1.5);count=6400
        return jsonify({'vertices':mesh.vertices.flatten().tolist(),'faces':mesh.faces.flatten().tolist(),'count':count})
    except Exception as e:
        import traceback;traceback.print_exc()
        return jsonify({'error':str(e)}),500

@app.route('/api/map/stl',methods=['POST'])
def api_map_stl():
    try:
        d=request.json;lat,lon,radius,size,height=d['lat'],d['lon'],d['radius'],d['size'],d['height']
        mode=d['mode'];lw=d.get('lineWidth',1.2)
        if mode=='streets':gdf=fetch_streets(lat,lon,radius);mesh=create_streets_mesh(gdf,lat,lon,radius,size,height,lw)
        elif mode=='city':mesh=create_city_mesh(lat,lon,radius,size,height,lw,1.0,1.5)
        else:mesh=create_terrain_mesh(lat,lon,radius,size,height*5,1.5)
        mesh.fill_holes();mesh.fix_normals();buf=io.BytesIO();mesh.export(buf,file_type='stl');buf.seek(0)
        return send_file(buf,mimetype='application/octet-stream',as_attachment=True,download_name='map.stl')
    except Exception as e:
        import traceback;traceback.print_exc()
        return jsonify({'error':str(e)}),500

@app.route('/api/image/convert',methods=['POST'])
def api_image_convert():
    try:
        d=request.json;img_b64=d['image']
        if ',' in img_b64:img_b64=img_b64.split(',')[1]
        img_data=base64.b64decode(img_b64)
        mode=d.get('mode','outline');th=d.get('threshold',128);bl=d.get('blur',1)
        si=d.get('simplify',2);sm=d.get('smooth',0);inv=d.get('invert',False);sl=d.get('singleLine',False)
        if mode=='filled':svg,p,w,h=image_to_filled_svg(img_data,th,bl,si,sm,inv)
        else:svg,p,w,h=image_to_svg(img_data,th,bl,si,sm,inv,mode,sl)
        return jsonify({'svg':svg,'paths':p,'width':w,'height':h})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/print/process',methods=['POST'])
def api_print_process():
    try:
        d=request.json;img_b64=d['image']
        if ',' in img_b64:img_b64=img_b64.split(',')[1]
        img_data=base64.b64decode(img_b64)
        gray=process_print_image(img_data,d.get('contrast',1.5),d.get('denoise',2))
        preview=create_hueforge_preview(gray,d.get('num_colors',2))
        instr=calculate_print_instructions(d.get('total_height',2.4),d.get('base_height',0.6),d.get('layer_height',0.12),d.get('num_colors',2))
        _,mw,mh=create_relief_stl(gray,d.get('width',100),d.get('total_height',2.4),d.get('base_height',0.6),d.get('border',2))
        return jsonify({'instructions':instr,'model_width':mw,'model_height':mh})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/print/stl',methods=['POST'])
def api_print_stl():
    try:
        d=request.json;img_b64=d['image']
        if ',' in img_b64:img_b64=img_b64.split(',')[1]
        gray=process_print_image(base64.b64decode(img_b64),d.get('contrast',1.5),d.get('denoise',2))
        mesh,_,_=create_relief_stl(gray,d.get('width',100),d.get('total_height',2.4),d.get('base_height',0.6),d.get('border',2))
        buf=io.BytesIO();mesh.export(buf,file_type='stl');buf.seek(0)
        return send_file(buf,mimetype='application/octet-stream',as_attachment=True,download_name='hueforge.stl')
    except Exception as e:
        return jsonify({'error':str(e)}),500



@app.route('/api/shadow/process', methods=['POST'])
def api_shadow_process():
    try:
        d = request.json or {}
        img_data = d.get('image', '')
        height_mm = float(d.get('height', 140))
        thickness = float(d.get('thickness', 2))
        threshold = int(d.get('threshold', 128))
        smooth = float(d.get('smooth', 0.8))
        cleanup = float(d.get('cleanup', 0.0))
        max_points = int(d.get('maxPoints', 3000))
        foot_x = float(d.get('footX', 0))
        foot_y = float(d.get('footY', 0))
        foot_h = float(d.get('footH', 4))
        foot_width = 57.969
        if ',' in img_data:img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        combined, preview_rings, foot_ring, width_mm = _shadow_extract_geometry(img=img,height_mm=height_mm,threshold=threshold,smooth=smooth,foot_x=foot_x,foot_y=foot_y,foot_h=foot_h,foot_width=foot_width,cleanup=cleanup,max_points=max_points)
        meshes = []
        polys = list(combined.geoms) if combined.geom_type == 'MultiPolygon' else [combined]
        for p in polys:
            try:meshes.append(trimesh.creation.extrude_polygon(p, thickness))
            except:pass
        if not meshes:return jsonify({'error': 'Mesh creation failed'}), 400
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        height_out = float(height_mm + (foot_h if foot_h and foot_h > 0 else 0))
        return jsonify({'vertices': mesh.vertices.flatten().tolist(),'faces': mesh.faces.flatten().tolist(),'width': width_mm,'height': height_out,'preview': {'rings': preview_rings,'foot': foot_ring}})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/shadow/stl', methods=['POST'])
def api_shadow_stl():
    try:
        d = request.json or {}
        img_data = d.get('image', '')
        height_mm = float(d.get('height', 140))
        thickness = float(d.get('thickness', 2))
        threshold = int(d.get('threshold', 128))
        smooth = float(d.get('smooth', 0.8))
        cleanup = float(d.get('cleanup', 0.0))
        max_points = int(d.get('maxPoints', 3000))
        foot_x = float(d.get('footX', 0))
        foot_y = float(d.get('footY', 0))
        foot_h = float(d.get('footH', 4))
        foot_width = 57.969
        if ',' in img_data:img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        combined, _, _, _ = _shadow_extract_geometry(img=img,height_mm=height_mm,threshold=threshold,smooth=smooth,foot_x=foot_x,foot_y=foot_y,foot_h=foot_h,foot_width=foot_width,cleanup=cleanup,max_points=max_points)
        meshes = []
        polys = list(combined.geoms) if combined.geom_type == 'MultiPolygon' else [combined]
        for p in polys:
            try:meshes.append(trimesh.creation.extrude_polygon(p, thickness))
            except:pass
        if not meshes:return jsonify({'error': 'Mesh failed'}), 400
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        mesh.fix_normals()
        buf = io.BytesIO();mesh.export(buf, file_type='stl');buf.seek(0)
        return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name='family_silhouette.stl')
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status':'ok','version':'2.6','tools':['map','image','print','imprint','shadow']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
