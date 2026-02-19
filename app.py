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
.header{text-align:center;margin-bottom:40px}.logo-box{display:inline-flex;align-items:center;justify-content:center;margin-bottom:16px}.logo-img{max-width:400px;width:100%;height:auto}
.creator{display:inline-flex;align-items:center;gap:12px;background:rgba(0,174,66,0.1);border:1px solid rgba(0,174,66,0.3);border-radius:50px;padding:8px 20px 8px 8px;margin-top:16px;text-decoration:none;color:#fff;transition:all .2s}.creator:hover{background:rgba(0,174,66,0.2);border-color:#00AE42}.creator-avatar{width:36px;height:36px;background:#00AE42;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px}.creator-info{text-align:left}.creator-name{font-size:13px;font-weight:600;color:#00AE42}.creator-link{font-size:11px;color:#888}
.tools{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;max-width:900px;width:100%;margin-bottom:40px}.tool{background:#151515;border:1px solid #2a2a2a;border-radius:16px;padding:32px 24px;text-decoration:none;color:#fff;transition:all .2s;position:relative;overflow:hidden}.tool:hover{border-color:#00AE42;transform:translateY(-4px);box-shadow:0 10px 40px rgba(0,0,0,0.3)}.tool-badge{position:absolute;top:16px;right:16px;background:#00AE42;color:#fff;font-size:9px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.5px}.tool-badge.vector{background:#9333EA}.tool-icon{font-size:48px;margin-bottom:16px;display:block}.tool h3{font-size:20px;font-weight:700;margin-bottom:8px}.tool p{font-size:13px;color:#888;line-height:1.5}
.support{background:linear-gradient(135deg,#1a1a1a,#151515);border:1px solid #2a2a2a;border-radius:16px;padding:24px 32px;max-width:900px;width:100%;display:flex;align-items:center;gap:24px;margin-bottom:30px}.support-icon{font-size:40px;flex-shrink:0}.support-text{flex:1}.support-text h4{font-size:15px;font-weight:600;margin-bottom:6px;color:#fff}.support-text p{font-size:12px;color:#888;line-height:1.5}.support-btn{background:#00AE42;color:#fff;text-decoration:none;padding:12px 24px;border-radius:8px;font-size:13px;font-weight:600;white-space:nowrap;transition:all .2s}.support-btn:hover{background:#00C94B;transform:scale(1.05)}.support-price{text-align:center;margin-right:8px}.support-price .amount{font-size:24px;font-weight:700;color:#00AE42}.support-price .period{font-size:10px;color:#888}
.footer{text-align:center;font-size:12px;color:#666;margin-top:auto;padding-top:20px}.footer a{color:#00AE42;text-decoration:none}.footer span{margin:0 8px}
@media(max-width:700px){.logo-box{flex-direction:column;gap:12px}.logo-text{text-align:center}.logo-text h1{font-size:28px}.tools{grid-template-columns:1fr}.support{flex-direction:column;text-align:center;gap:16px}.support-price{margin:0}}
</style></head><body>
<div class="header"><div class="logo-box"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAAEKCAYAAAA8QgPpAACAK0lEQVR42uy9d3wcxf3//5qZ3evqXVaxZcmyJfeODcgF04zpJ3on9JIAISQETpdACIEklBDAhBBqQAeEjrGNLRkwBlvulpss25Jt9X51d2fm98edjOy40fL9fZJ9Ph4H1t3u7Ozszrzn/X7PvN+AiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYnJfyHEbAITExMTcyA9+H5M4WBiYmJi8uOQl5eXZbaCiYkJAFCzCb7bbL2kpCTzv6z9rAAcx3Bc2Hz8JiYmpgD5Hui67vhvuA+Px6MAwK2/u+r+O/78k2oAkFIe9r1oaGjoMp++iYmJiQkQ83tcftepQy+9e96cgd8d6XgTExMTczD4fm0nAVB4gLJ/0+bKfvQKzABQ9Z3Prj6grtXeagMAyjxQUPVN4dXpkCiBhBcydr8mJiYmJt8ZD6hnaZkipaT/KxKYEAIpPbTMU6bANH2amJiYGsgBmsQxCQ5ZISUhZODxuQWnZWZo0EbFJ6oAAMoAlTIIABAclDKw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" alt="SebGE Tools" class="logo-img"></div>
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
