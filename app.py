"""
SebGE Tools v1.0
Created by SebGE - https://makerworld.com/de/@SebGE

Tools:
1. Map→STL - 3D printable maps
2. Image→SVG - Vectorize images
"""
import os,io,requests,zipfile,base64,subprocess,tempfile
from flask import Flask,request,jsonify,send_file,render_template_string
from flask_cors import CORS
import geopandas as gpd
import numpy as np
from shapely.geometry import box,MultiPolygon,Polygon,LineString
from shapely.ops import unary_union
from shapely.affinity import translate
import trimesh
from PIL import Image
import cv2

app=Flask(__name__)
CORS(app)

# ==================== SHARED ====================
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

STREET_TYPES={
    'motorway':'motorway|motorway_link','trunk':'trunk|trunk_link','primary':'primary|primary_link',
    'secondary':'secondary|secondary_link','tertiary':'tertiary|tertiary_link','residential':'residential|living_street',
    'unclassified':'unclassified','service':'service','pedestrian':'pedestrian','footway':'footway|steps',
    'cycleway':'cycleway','track':'track','bridleway':'bridleway','path':'path'
}

# ==================== MAP FUNCTIONS ====================
def geocode(q):
    known={'berlin':(52.52,13.405),'düsseldorf':(51.2277,6.7735),'köln':(50.9375,6.9603),'münchen':(48.1351,11.582),'hamburg':(53.5511,9.9937),'frankfurt':(50.1109,8.6821),'paris':(48.8566,2.3522),'new york':(40.7128,-74.006),'manhattan':(40.7831,-73.9712),'london':(51.5074,-0.1278),'tokyo':(35.6762,139.6503),'alps':(46.8,10.5),'matterhorn':(45.9763,7.6586)}
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
        r=requests.get('https://nominatim.openstreetmap.org/search',params={'q':q,'format':'json','limit':1},headers={'User-Agent':'SebGETools/1.0'},timeout=10)
        if r.ok and r.json():d=r.json()[0];return{'lat':float(d['lat']),'lon':float(d['lon']),'name':d.get('display_name',q)[:50]}
    except:pass
    return None

def fetch_streets(lat,lon,radius,types=None):
    if types is None:types=['motorway','trunk','primary','secondary','tertiary','residential']
    tags=[STREET_TYPES[t] for t in types if t in STREET_TYPES]
    if not tags:tags=['residential']
    regex='|'.join(tags);bbox=radius/111000
    q=f'[out:json][timeout:60];(way["highway"~"^({regex})$"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5}););out body;>;out skel qt;'
    try:
        r=requests.post('https://overpass-api.de/api/interpreter',data={'data':q},timeout=60)
        if not r.ok:raise Exception("API error")
        data=r.json();nodes={e['id']:(e['lon'],e['lat']) for e in data.get('elements',[]) if e['type']=='node'}
        lines=[]
        for e in data.get('elements',[]):
            if e['type']=='way' and 'nodes' in e:
                coords=[nodes[n] for n in e['nodes'] if n in nodes]
                if len(coords)>=2:lines.append(LineString(coords))
        if lines:return gpd.GeoDataFrame(geometry=lines,crs='EPSG:4326')
    except Exception as ex:print(f"Streets error: {ex}")
    import random;random.seed(int(abs(lat*1000+lon*100)));dl=radius/111320;dlo=dl/np.cos(np.radians(lat));lines=[]
    for i in range(6):
        y=lat-dl+(2*dl*i/5);lines.append(LineString([(lon-dlo+(2*dlo*j/20),y+random.uniform(-dl*.02,dl*.02)) for j in range(21)]))
        x=lon-dlo+(2*dlo*i/5);lines.append(LineString([(x+random.uniform(-dlo*.02,dlo*.02),lat-dl+(2*dl*j/20)) for j in range(21)]))
    return gpd.GeoDataFrame(geometry=lines,crs='EPSG:4326')

def fetch_buildings(lat,lon,radius):
    bbox=radius/111000
    q=f'[out:json][timeout:60];way["building"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5});out body;>;out skel qt;'
    try:
        r=requests.post('https://overpass-api.de/api/interpreter',data={'data':q},timeout=60)
        if not r.ok:return[]
        data=r.json();nodes={e['id']:(e['lon'],e['lat']) for e in data.get('elements',[]) if e['type']=='node'}
        buildings=[]
        for e in data.get('elements',[]):
            if e['type']=='way' and 'nodes' in e:
                coords=[nodes[n] for n in e['nodes'] if n in nodes]
                if len(coords)>=4:
                    try:
                        poly=Polygon(coords)
                        if poly.is_valid and poly.area>0:
                            tags=e.get('tags',{});h=10
                            if 'height' in tags:
                                try:h=float(tags['height'].replace('m',''))
                                except:pass
                            elif 'building:levels' in tags:
                                try:h=float(tags['building:levels'])*3
                                except:pass
                            buildings.append({'geometry':poly,'height':h})
                    except:pass
        return buildings
    except:return[]

def fetch_water(lat,lon,radius):
    bbox=radius/111000
    q=f'[out:json][timeout:60];(way["natural"="water"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5});way["waterway"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5}););out body;>;out skel qt;'
    try:
        r=requests.post('https://overpass-api.de/api/interpreter',data={'data':q},timeout=60)
        if not r.ok:return[]
        data=r.json();nodes={e['id']:(e['lon'],e['lat']) for e in data.get('elements',[]) if e['type']=='node'}
        water=[]
        for e in data.get('elements',[]):
            if e['type']=='way' and 'nodes' in e:
                coords=[nodes[n] for n in e['nodes'] if n in nodes]
                if len(coords)>=3:
                    try:poly=Polygon(coords);water.append(poly) if poly.is_valid else None
                    except:pass
        return water
    except:return[]

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
    gdf=fetch_streets(lat,lon,radius);buildings=fetch_buildings(lat,lon,radius);water=fetch_water(lat,lon,radius)
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
    for w in water:
        try:
            wg=gpd.GeoDataFrame(geometry=[w],crs='EPSG:4326').to_crs(utm).geometry.iloc[0]
            c=np.array(wg.exterior.coords);c[:,0]=(c[:,0]-ctr.x)*sc;c[:,1]=(c[:,1]-ctr.y)*sc
            wp=Polygon(c.tolist()).intersection(box(-size/2,-size/2,size/2,size/2))
            if not wp.is_empty:meshes.append(trimesh.creation.extrude_polygon(wp if wp.geom_type=='Polygon' else list(wp.geoms)[0],base_h*0.5))
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

# ==================== IMAGE TO SVG ====================
def image_to_svg(img_data, threshold=128, blur=1, simplify=2, invert=False, mode='outline'):
    """Convert image to SVG using OpenCV edge detection and contour tracing."""
    # Decode image
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply blur if specified
    if blur > 0:
        blur_size = blur * 2 + 1
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Invert if needed
    if invert:
        gray = 255 - gray
    
    if mode == 'outline':
        # Edge detection
        edges = cv2.Canny(gray, threshold * 0.5, threshold)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    elif mode == 'threshold':
        # Simple threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    elif mode == 'adaptive':
        # Adaptive threshold for varying lighting
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:  # posterize
        # Reduce colors then trace
        levels = max(2, min(8, threshold // 32))
        quantized = (gray // (256 // levels)) * (256 // levels)
        edges = cv2.Canny(quantized, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Build SVG
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
    svg += f'<rect width="100%" height="100%" fill="white"/>\n'
    
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Simplify contour
        epsilon = simplify
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
        
        # Build path
        points = approx.reshape(-1, 2)
        d = f"M {points[0][0]},{points[0][1]}"
        for p in points[1:]:
            d += f" L {p[0]},{p[1]}"
        d += " Z"
        paths.append(d)
    
    # Combine paths
    if paths:
        svg += f'<path d="{" ".join(paths)}" fill="none" stroke="black" stroke-width="1"/>\n'
    
    svg += '</svg>'
    return svg, len(paths), w, h

def image_to_filled_svg(img_data, threshold=128, blur=1, simplify=2, invert=False):
    """Create filled SVG (silhouette style)."""
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (blur * 2 + 1, blur * 2 + 1), 0)
    
    if invert:
        gray = 255 - gray
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
    svg += f'<rect width="100%" height="100%" fill="white"/>\n'
    
    paths = []
    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue
        
        approx = cv2.approxPolyDP(contour, simplify, True)
        if len(approx) < 3:
            continue
        
        points = approx.reshape(-1, 2)
        d = f"M {points[0][0]},{points[0][1]}"
        for p in points[1:]:
            d += f" L {p[0]},{p[1]}"
        d += " Z"
        paths.append(d)
    
    if paths:
        svg += f'<path d="{" ".join(paths)}" fill="black" stroke="none" fill-rule="evenodd"/>\n'
    
    svg += '</svg>'
    return svg, len(paths), w, h

# ==================== HTML TEMPLATES ====================
LANDING_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SebGE Tools | 3D Print & Design</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--g:#00AE42;--gd:#009639;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--t);min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px}
.logo{display:flex;align-items:center;gap:16px;margin-bottom:12px}
.logo-icon{width:64px;height:64px;background:linear-gradient(135deg,var(--g),var(--gd));border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:32px;box-shadow:0 8px 32px rgba(0,174,66,.3)}
.logo h1{font-size:32px;font-weight:800}
.logo small{display:block;font-size:12px;color:var(--g);margin-top:4px}
.creator{display:flex;align-items:center;gap:10px;background:var(--c);border:1px solid var(--g);border-radius:30px;padding:8px 20px 8px 8px;text-decoration:none;margin-bottom:40px;transition:all .2s}
.creator:hover{background:var(--c2);transform:translateY(-2px)}
.cr-av{width:32px;height:32px;background:var(--g);border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px}
.cr-info{font-size:12px}
.cr-name{font-weight:600}
.cr-link{color:var(--g);font-size:10px}
.tools{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px;max-width:800px;width:100%}
.tool{background:var(--c);border:2px solid var(--br);border-radius:20px;padding:32px;text-align:center;text-decoration:none;color:var(--t);transition:all .2s;position:relative;overflow:hidden}
.tool::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(135deg,rgba(0,174,66,.1),transparent);opacity:0;transition:opacity .2s}
.tool:hover{border-color:var(--g);transform:translateY(-4px);box-shadow:0 12px 40px rgba(0,174,66,.2)}
.tool:hover::before{opacity:1}
.tool-icon{font-size:56px;margin-bottom:16px;display:block}
.tool-name{font-size:20px;font-weight:700;margin-bottom:8px}
.tool-desc{font-size:13px;color:var(--t2);line-height:1.5}
.tool-badge{position:absolute;top:16px;right:16px;background:var(--g);color:#fff;font-size:9px;padding:4px 10px;border-radius:20px;font-weight:700}
.footer{margin-top:60px;text-align:center;color:var(--t2);font-size:11px}
.footer a{color:var(--g);text-decoration:none}
@media(max-width:600px){.tools{grid-template-columns:1fr}.logo h1{font-size:24px}}
</style></head><body>
<div class="logo">
<div class="logo-icon">🛠️</div>
<div><h1>SebGE Tools</h1><small>3D PRINT & DESIGN TOOLS</small></div>
</div>
<a href="https://makerworld.com/de/@SebGE" target="_blank" class="creator">
<div class="cr-av">S</div>
<div class="cr-info"><div class="cr-name">Created by SebGE</div><div class="cr-link">MakerWorld Profile →</div></div>
</a>
<div class="tools">
<a href="/map" class="tool">
<span class="tool-badge">3D PRINT</span>
<span class="tool-icon">🗺️</span>
<div class="tool-name">Map → STL</div>
<div class="tool-desc">Create 3D printable maps from any location. Streets, cities, or terrain with real elevation data.</div>
</a>
<a href="/image" class="tool">
<span class="tool-badge">VECTOR</span>
<span class="tool-icon">🎨</span>
<div class="tool-name">Image → SVG</div>
<div class="tool-desc">Convert images to vector graphics. Perfect for laser cutting, vinyl cutting, or clean scaling.</div>
</a>
</div>
<div class="footer">Made with ❤️ by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a> • Open Source Tools for Makers</div>
</body></html>'''

IMAGE_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Image→SVG | SebGE Tools</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--g:#00AE42;--gd:#009639;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--t);min-height:100vh}
.app{display:grid;grid-template-columns:280px 1fr 280px;height:100vh}
.panel{background:var(--c);padding:16px;display:flex;flex-direction:column;gap:12px;overflow-y:auto}
.panel::-webkit-scrollbar{width:5px}
.panel::-webkit-scrollbar-thumb{background:var(--br);border-radius:3px}

.logo{display:flex;align-items:center;gap:10px;padding-bottom:12px;border-bottom:1px solid var(--br)}
.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--g),var(--gd));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px}
.logo h1{font-size:14px;font-weight:700}
.logo small{display:block;font-size:8px;color:var(--g);margin-top:2px}
.back{display:inline-flex;align-items:center;gap:6px;color:var(--t2);text-decoration:none;font-size:10px;margin-bottom:8px}
.back:hover{color:var(--g)}

.upload-area{border:2px dashed var(--br);border-radius:12px;padding:32px 16px;text-align:center;cursor:pointer;transition:all .2s}
.upload-area:hover{border-color:var(--g);background:rgba(0,174,66,.05)}
.upload-area.dragover{border-color:var(--g);background:rgba(0,174,66,.1)}
.upload-icon{font-size:40px;margin-bottom:8px}
.upload-text{font-size:11px;color:var(--t2)}
.upload-text strong{color:var(--t);display:block;margin-bottom:4px}
#fileInput{display:none}

.sec{background:var(--c2);border-radius:8px;padding:12px}
.sec-title{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--g);margin-bottom:10px}

.modes{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}
.mode{padding:10px 8px;background:var(--c);border:2px solid var(--br);border-radius:6px;cursor:pointer;text-align:center;transition:all .15s}
.mode:hover{border-color:var(--g)}
.mode.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.mode .icon{font-size:18px}
.mode .name{font-size:9px;font-weight:600;margin-top:4px}

.slider{margin-bottom:8px}
.slider-head{display:flex;justify-content:space-between;margin-bottom:3px}
.slider label{font-size:10px;color:var(--t2)}
.slider .val{font-size:10px;color:var(--g);font-family:monospace;font-weight:600}
.slider input{width:100%;height:4px;background:var(--c);border-radius:2px;-webkit-appearance:none}
.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--g);border-radius:50%;cursor:pointer}

.checkbox{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:11px;padding:8px 0}
.checkbox input{display:none}
.checkbox .box{width:18px;height:18px;border:2px solid var(--br);border-radius:4px;display:flex;align-items:center;justify-content:center;transition:all .15s}
.checkbox input:checked+.box{background:var(--g);border-color:var(--g)}
.checkbox .box::after{content:'✓';color:#fff;font-size:11px;opacity:0}
.checkbox input:checked+.box::after{opacity:1}

.btn{padding:12px;border:none;border-radius:8px;font-size:11px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;transition:all .15s;width:100%}
.btn-primary{background:linear-gradient(135deg,var(--g),var(--gd));color:#fff}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,174,66,.3)}
.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}
.btn-secondary:hover{border-color:var(--g);color:var(--g)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

.preview-container{background:var(--c);display:flex;align-items:center;justify-content:center;position:relative}
.preview-box{background:#fff;border-radius:8px;box-shadow:0 4px 24px rgba(0,0,0,.5);max-width:90%;max-height:90%;overflow:hidden;display:flex;align-items:center;justify-content:center}
.preview-box img,.preview-box svg{max-width:100%;max-height:70vh;display:block}
.preview-placeholder{color:var(--t2);font-size:12px;text-align:center}
.preview-placeholder .icon{font-size:48px;margin-bottom:12px;opacity:.5}
.preview-tabs{position:absolute;top:16px;left:50%;transform:translateX(-50%);display:flex;gap:4px;background:var(--c);border-radius:20px;padding:4px}
.preview-tab{padding:6px 16px;border-radius:16px;font-size:10px;font-weight:600;cursor:pointer;transition:all .15s;color:var(--t2)}
.preview-tab:hover{color:var(--t)}
.preview-tab.active{background:var(--g);color:#fff}

.stats{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.stat{background:var(--c2);border-radius:6px;padding:8px;text-align:center}
.stat label{font-size:8px;color:var(--t2);text-transform:uppercase}
.stat .val{font-size:12px;color:var(--g);font-family:monospace;font-weight:600;margin-top:2px}

.status{padding:8px;border-radius:6px;font-size:10px;display:none;text-align:center}
.status.error{display:block;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#ef4444}
.status.success{display:block;background:rgba(0,174,66,.1);border:1px solid rgba(0,174,66,.3);color:var(--g)}

.footer{margin-top:auto;text-align:center;font-size:8px;color:var(--t2);padding-top:10px;border-top:1px solid var(--br)}
.footer a{color:var(--g);text-decoration:none}

.spinner{width:14px;height:14px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

@media(max-width:900px){.app{grid-template-columns:1fr}.panel{display:none}}
</style></head><body>
<div class="app">
<div class="panel">
<a href="/" class="back">← Back to Tools</a>
<div class="logo">
<div class="logo-icon">🎨</div>
<div><h1>Image → SVG</h1><small>VECTORIZE IMAGES</small></div>
</div>

<div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
<div class="upload-icon">📁</div>
<div class="upload-text"><strong>Drop image here</strong>or click to browse<br>PNG, JPG, WEBP</div>
</div>
<input type="file" id="fileInput" accept="image/*">

<div class="sec">
<div class="sec-title">Trace Mode</div>
<div class="modes">
<div class="mode active" data-mode="outline" onclick="setMode('outline')"><div class="icon">✏️</div><div class="name">Outline</div></div>
<div class="mode" data-mode="filled" onclick="setMode('filled')"><div class="icon">⬛</div><div class="name">Filled</div></div>
<div class="mode" data-mode="threshold" onclick="setMode('threshold')"><div class="icon">🔲</div><div class="name">Threshold</div></div>
<div class="mode" data-mode="posterize" onclick="setMode('posterize')"><div class="icon">🎨</div><div class="name">Posterize</div></div>
</div>
</div>

<div class="sec">
<div class="sec-title">Settings</div>
<div class="slider"><div class="slider-head"><label>Threshold</label><span class="val" id="threshV">128</span></div>
<input type="range" id="thresh" min="10" max="245" value="128" oninput="$('threshV').textContent=this.value"></div>
<div class="slider"><div class="slider-head"><label>Blur</label><span class="val" id="blurV">1</span></div>
<input type="range" id="blur" min="0" max="5" value="1" oninput="$('blurV').textContent=this.value"></div>
<div class="slider"><div class="slider-head"><label>Simplify</label><span class="val" id="simpV">2</span></div>
<input type="range" id="simp" min="1" max="10" value="2" oninput="$('simpV').textContent=this.value"></div>
<label class="checkbox"><input type="checkbox" id="invert"><span class="box"></span>Invert colors</label>
</div>

<button class="btn btn-secondary" id="btnPreview" onclick="convert()" disabled>👁 Preview</button>
<button class="btn btn-primary" id="btnExport" onclick="exportSVG()" disabled>⬇ Download SVG</button>

<div class="status" id="status"></div>

<div class="footer">Made with ❤️ by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div>
</div>

<div class="preview-container" id="previewContainer">
<div class="preview-tabs">
<div class="preview-tab active" data-view="svg" onclick="showView('svg')">SVG Result</div>
<div class="preview-tab" data-view="original" onclick="showView('original')">Original</div>
</div>
<div class="preview-placeholder" id="placeholder">
<div class="icon">🖼️</div>
Upload an image to start
</div>
<div class="preview-box" id="previewBox" style="display:none"></div>
</div>

<div class="panel">
<div class="sec-title">Output Info</div>
<div class="stats">
<div class="stat"><label>Paths</label><div class="val" id="statPaths">—</div></div>
<div class="stat"><label>Size</label><div class="val" id="statSize">—</div></div>
</div>

<div class="sec" style="margin-top:12px">
<div class="sec-title">Tips</div>
<div style="font-size:9px;color:var(--t2);line-height:1.5">
<strong style="color:var(--g)">Outline:</strong> Best for drawings & logos<br>
<strong style="color:var(--g)">Filled:</strong> Creates silhouettes<br>
<strong style="color:var(--g)">Threshold:</strong> High contrast B/W<br>
<strong style="color:var(--g)">Posterize:</strong> Reduces detail levels<br><br>
Higher blur = smoother lines<br>
Higher simplify = fewer points
</div>
</div>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
let mode='outline',imageData=null,svgResult=null,originalUrl=null;

// Drag & drop
const ua=$('uploadArea');
['dragenter','dragover'].forEach(e=>ua.addEventListener(e,ev=>{ev.preventDefault();ua.classList.add('dragover')}));
['dragleave','drop'].forEach(e=>ua.addEventListener(e,ev=>{ev.preventDefault();ua.classList.remove('dragover')}));
ua.addEventListener('drop',e=>{if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])});
$('fileInput').addEventListener('change',e=>{if(e.target.files.length)handleFile(e.target.files[0])});

function handleFile(file){
    if(!file.type.startsWith('image/')){msg('error','Please upload an image');return}
    const reader=new FileReader();
    reader.onload=e=>{
        imageData=e.target.result;
        originalUrl=URL.createObjectURL(file);
        $('btnPreview').disabled=false;
        $('placeholder').style.display='none';
        $('previewBox').style.display='flex';
        $('previewBox').innerHTML=`<img src="${originalUrl}">`;
        msg('success','✓ Image loaded: '+file.name);
    };
    reader.readAsDataURL(file);
}

function setMode(m){
    mode=m;
    document.querySelectorAll('.mode').forEach(el=>el.classList.toggle('active',el.dataset.mode===m));
}

function showView(view){
    document.querySelectorAll('.preview-tab').forEach(t=>t.classList.toggle('active',t.dataset.view===view));
    if(view==='original'&&originalUrl){
        $('previewBox').innerHTML=`<img src="${originalUrl}">`;
    }else if(view==='svg'&&svgResult){
        $('previewBox').innerHTML=svgResult;
    }
}

async function convert(){
    if(!imageData)return;
    const btn=$('btnPreview');
    btn.disabled=true;btn.innerHTML='<div class="spinner"></div>';
    try{
        const r=await fetch('/api/image/convert',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                image:imageData,
                mode,
                threshold:+$('thresh').value,
                blur:+$('blur').value,
                simplify:+$('simp').value,
                invert:$('invert').checked
            })
        });
        const d=await r.json();
        if(d.error)throw new Error(d.error);
        svgResult=d.svg;
        $('previewBox').innerHTML=svgResult;
        $('statPaths').textContent=d.paths;
        $('statSize').textContent=d.width+'×'+d.height;
        $('btnExport').disabled=false;
        showView('svg');
        msg('success','✓ Converted! '+d.paths+' paths');
    }catch(e){msg('error',e.message)}
    finally{btn.disabled=false;btn.innerHTML='👁 Preview'}
}

function exportSVG(){
    if(!svgResult)return;
    const blob=new Blob([svgResult],{type:'image/svg+xml'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='converted.svg';
    a.click();
    msg('success','✓ SVG downloaded!');
}

function msg(type,text){
    const el=$('status');
    el.className='status '+type;
    el.textContent=text;
    setTimeout(()=>el.className='status',4000);
}
</script>
</body></html>'''

# Map HTML (same as before but with back link)
MAP_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Map→STL | SebGE Tools</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--g:#00AE42;--gd:#009639;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--t);height:100vh;overflow:hidden}
.app{display:grid;grid-template-columns:300px 1fr 280px;height:100vh}
.panel{background:var(--c);padding:14px;display:flex;flex-direction:column;gap:10px;overflow-y:auto}
.panel::-webkit-scrollbar{width:5px}
.panel::-webkit-scrollbar-thumb{background:var(--br);border-radius:3px}
.back{display:inline-flex;align-items:center;gap:6px;color:var(--t2);text-decoration:none;font-size:10px;margin-bottom:4px}
.back:hover{color:var(--g)}
.logo{display:flex;align-items:center;gap:10px;padding-bottom:10px;border-bottom:1px solid var(--br)}
.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--g),var(--gd));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px}
.logo h1{font-size:14px;font-weight:700}
.logo small{display:block;font-size:8px;color:var(--g);font-weight:500;margin-top:2px}
.search{position:relative}
.search input{width:100%;padding:8px 10px 8px 30px;background:var(--c2);border:1px solid var(--br);border-radius:6px;color:var(--t);font-size:11px}
.search input:focus{outline:none;border-color:var(--g)}
.search svg{position:absolute;left:8px;top:50%;transform:translateY(-50%);color:var(--t2);width:14px;height:14px}
.coords{display:flex;gap:6px;margin-top:4px}
.coord{flex:1;background:var(--c2);border-radius:4px;padding:4px 6px;font-size:9px}
.coord label{color:var(--t2);font-size:7px;text-transform:uppercase}
.coord span{color:var(--g);font-family:monospace;font-weight:600}
.sec{background:var(--c2);border-radius:8px;padding:10px}
.sec-title{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--g);margin-bottom:8px}
.modes{display:flex;gap:4px}
.mode{flex:1;padding:8px 4px;background:var(--c);border:2px solid var(--br);border-radius:6px;cursor:pointer;text-align:center;transition:all .15s}
.mode:hover{border-color:var(--g)}
.mode.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.mode .icon{font-size:16px}
.mode .name{font-size:8px;font-weight:600;margin-top:2px}
.mode .desc{font-size:7px;color:var(--t2)}
.presets{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px}
.preset{padding:4px 8px;background:var(--c);border:1px solid var(--br);border-radius:4px;cursor:pointer;font-size:8px;transition:all .15s}
.preset:hover{border-color:var(--g);color:var(--g)}
.preset.active{background:var(--g);border-color:var(--g);color:#fff}
.street-types{display:grid;grid-template-columns:repeat(2,1fr);gap:3px}
.stype{display:flex;align-items:center;gap:4px;padding:4px 6px;background:var(--c);border:1px solid var(--br);border-radius:4px;cursor:pointer;font-size:8px;transition:all .15s}
.stype:hover{border-color:var(--g)}
.stype.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.stype .icon{font-size:10px}
.stype .check{width:10px;height:10px;border:1px solid var(--br);border-radius:2px;display:flex;align-items:center;justify-content:center;font-size:7px;margin-left:auto}
.stype.active .check{background:var(--g);border-color:var(--g);color:#fff}
.markers{display:grid;grid-template-columns:repeat(5,1fr);gap:3px}
.marker{padding:5px 2px;background:var(--c);border:2px solid var(--br);border-radius:5px;cursor:pointer;text-align:center;font-size:12px;transition:all .15s}
.marker:hover{border-color:var(--g)}
.marker.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.marker span{display:block;font-size:7px;margin-top:1px}
.slider{margin-bottom:5px}
.slider-head{display:flex;justify-content:space-between;margin-bottom:2px}
.slider label{font-size:9px;color:var(--t2)}
.slider .val{font-size:9px;color:var(--g);font-family:monospace;font-weight:600}
.slider input{width:100%;height:4px;background:var(--c);border-radius:2px;-webkit-appearance:none}
.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;background:var(--g);border-radius:50%;cursor:pointer}
.btn{padding:10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:5px;transition:all .15s}
.btn-primary{background:linear-gradient(135deg,var(--g),var(--gd));color:#fff}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,174,66,.3)}
.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}
.btn-secondary:hover{border-color:var(--g);color:var(--g)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.status{padding:6px;border-radius:5px;font-size:9px;display:none;text-align:center}
.status.error{display:block;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#ef4444}
.status.success{display:block;background:rgba(0,174,66,.1);border:1px solid rgba(0,174,66,.3);color:var(--g)}
.map-container{position:relative;background:var(--c)}
#map{width:100%;height:100%}
.map-overlay{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.8);padding:5px 12px;border-radius:15px;font-size:8px;color:var(--t2)}
.preview-title{font-size:11px;font-weight:600;display:flex;align-items:center;gap:6px}
.preview-title .badge{font-size:7px;background:var(--g);color:#fff;padding:2px 5px;border-radius:8px}
#preview3d{flex:1;background:var(--c2);border-radius:8px;min-height:180px;display:flex;align-items:center;justify-content:center;color:var(--t2);font-size:10px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.stat{background:var(--c2);border-radius:5px;padding:5px;text-align:center}
.stat label{font-size:7px;color:var(--t2);text-transform:uppercase}
.stat .val{font-size:10px;color:var(--g);font-family:monospace;font-weight:600;margin-top:1px}
.info{font-size:8px;color:var(--t2);line-height:1.4;background:var(--c2);border-radius:5px;padding:8px}
.info strong{color:var(--g)}
.footer{margin-top:auto;text-align:center;font-size:8px;color:var(--t2);padding-top:8px;border-top:1px solid var(--br)}
.footer a{color:var(--g);text-decoration:none}
.spinner{width:12px;height:12px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.marker-opts{margin-top:6px;padding-top:6px;border-top:1px solid var(--br)}
.street-sec{display:block}
@media(max-width:900px){.app{grid-template-columns:1fr}.panel{display:none}}
</style></head><body>
<div class="app">
<div class="panel">
<a href="/" class="back">← Back to Tools</a>
<div class="logo">
<div class="logo-icon">🗺️</div>
<div><h1>Map → STL</h1><small>3D PRINT YOUR WORLD</small></div>
</div>
<div class="search">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
<input type="text" id="loc" placeholder="Search location..." value="Düsseldorf" onkeypress="if(event.key==='Enter')search()">
</div>
<div class="coords">
<div class="coord"><label>LAT</label><br><span id="latV">51.2277</span></div>
<div class="coord"><label>LON</label><br><span id="lonV">6.7735</span></div>
</div>
<div class="sec">
<div class="sec-title">Mode</div>
<div class="modes">
<div class="mode active" data-mode="streets" onclick="setMode('streets')"><div class="icon">🛣️</div><div class="name">Streets</div><div class="desc">Roads only</div></div>
<div class="mode" data-mode="city" onclick="setMode('city')"><div class="icon">🏙️</div><div class="name">City</div><div class="desc">+Buildings</div></div>
<div class="mode" data-mode="terrain" onclick="setMode('terrain')"><div class="icon">🏔️</div><div class="name">Terrain</div><div class="desc">+Elevation</div></div>
</div>
</div>
<div class="sec street-sec" id="streetSec">
<div class="sec-title">Street Types</div>
<div class="presets">
<div class="preset active" onclick="preset('city')">🏙️ City</div>
<div class="preset" onclick="preset('rural')">🌾 Rural</div>
<div class="preset" onclick="preset('all')">📍 All</div>
<div class="preset" onclick="preset('paths')">🚶 Paths</div>
</div>
<div class="street-types">
<div class="stype active" data-t="motorway"><span class="icon">🚀</span>Highway<span class="check">✓</span></div>
<div class="stype active" data-t="trunk"><span class="icon">🛣️</span>Trunk<span class="check">✓</span></div>
<div class="stype active" data-t="primary"><span class="icon">🔴</span>Primary<span class="check">✓</span></div>
<div class="stype active" data-t="secondary"><span class="icon">🟠</span>Secondary<span class="check">✓</span></div>
<div class="stype active" data-t="tertiary"><span class="icon">🟡</span>Tertiary<span class="check">✓</span></div>
<div class="stype active" data-t="residential"><span class="icon">🏠</span>Residential<span class="check">✓</span></div>
<div class="stype" data-t="unclassified"><span class="icon">📍</span>Unclassified<span class="check">✓</span></div>
<div class="stype" data-t="service"><span class="icon">🅿️</span>Service<span class="check">✓</span></div>
<div class="stype" data-t="track"><span class="icon">🌾</span>Field Track<span class="check">✓</span></div>
<div class="stype" data-t="path"><span class="icon">🥾</span>Path<span class="check">✓</span></div>
<div class="stype" data-t="footway"><span class="icon">🚶</span>Footway<span class="check">✓</span></div>
<div class="stype" data-t="cycleway"><span class="icon">🚴</span>Cycleway<span class="check">✓</span></div>
<div class="stype" data-t="bridleway"><span class="icon">🐴</span>Bridleway<span class="check">✓</span></div>
<div class="stype" data-t="pedestrian"><span class="icon">🚶‍♂️</span>Pedestrian<span class="check">✓</span></div>
</div>
</div>
<div class="sec street-sec" id="markerSec">
<div class="sec-title">Center Marker</div>
<div class="markers">
<div class="marker active" data-m="none" onclick="setMarker('none')">❌<span>None</span></div>
<div class="marker" data-m="heart" onclick="setMarker('heart')">❤️<span>Heart</span></div>
<div class="marker" data-m="star" onclick="setMarker('star')">⭐<span>Star</span></div>
<div class="marker" data-m="pin" onclick="setMarker('pin')">📍<span>Pin</span></div>
<div class="marker" data-m="circle" onclick="setMarker('circle')">⭕<span>Circle</span></div>
</div>
<div class="marker-opts" id="markerOpts" style="display:none">
<div class="slider"><div class="slider-head"><label>Marker Size</label><span class="val" id="msV">12mm</span></div>
<input type="range" id="ms" min="5" max="25" value="12" step="1" oninput="$('msV').textContent=this.value+'mm'"></div>
<div class="slider"><div class="slider-head"><label>Gap (Frame)</label><span class="val" id="mgV">2mm</span></div>
<input type="range" id="mg" min="0" max="5" value="2" step="0.5" oninput="$('mgV').textContent=this.value+'mm'"></div>
</div>
</div>
<div class="sec">
<div class="sec-title">Settings</div>
<div class="slider"><div class="slider-head"><label>Radius</label><span class="val" id="radV">500m</span></div>
<input type="range" id="rad" min="100" max="3000" value="500" step="50" oninput="updRad()"></div>
<div class="slider"><div class="slider-head"><label>Model Size</label><span class="val" id="sizeV">80mm</span></div>
<input type="range" id="size" min="30" max="200" value="80" step="5" oninput="$('sizeV').textContent=this.value+'mm'"></div>
<div class="slider" id="lwSlider"><div class="slider-head"><label>Line Width</label><span class="val" id="lwV">1.2mm</span></div>
<input type="range" id="lw" min="0.4" max="3" value="1.2" step="0.1" oninput="$('lwV').textContent=this.value+'mm'"></div>
<div class="slider"><div class="slider-head"><label>Height</label><span class="val" id="htV">2mm</span></div>
<input type="range" id="ht" min="1" max="10" value="2" step="0.5" oninput="$('htV').textContent=this.value+'mm'"></div>
</div>
</div>
<div class="map-container"><div id="map"></div><div class="map-overlay">Click map to set center</div></div>
<div class="panel">
<div class="preview-title">3D Preview <span class="badge" id="badge" style="display:none">Ready</span></div>
<div id="preview3d">Click Preview to load</div>
<div class="stats">
<div class="stat"><label>Mode</label><div class="val" id="statMode">Streets</div></div>
<div class="stat"><label>Data</label><div class="val" id="statData">—</div></div>
</div>
<div class="btns">
<button class="btn btn-secondary" id="btnPreview" onclick="preview()">👁 Preview</button>
<button class="btn btn-primary" id="btnExport" onclick="exportSTL()">⬇ Export STL</button>
</div>
<div class="status" id="status"></div>
<div class="info"><strong>🎨 Multi-Color:</strong> Use Gap to separate marker from streets for different colors in Bambu Studio!</div>
<div class="footer">Made with ❤️ by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div>
</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const $=id=>document.getElementById(id);
let map,mapMarker,circle,lat=51.2277,lon=6.7735,mode='streets',markerType='none';
let scene,camera,renderer,mesh;
const presets={city:['motorway','trunk','primary','secondary','tertiary','residential'],rural:['primary','secondary','tertiary','residential','track','unclassified','path','service'],all:['motorway','trunk','primary','secondary','tertiary','residential','unclassified','service','track','footway','cycleway','path','pedestrian','bridleway'],paths:['footway','cycleway','path','pedestrian','bridleway','track']};
function init(){
    map=L.map('map',{zoomControl:false}).setView([lat,lon],14);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{maxZoom:19}).addTo(map);
    updateMap();map.on('click',e=>{lat=e.latlng.lat;lon=e.latlng.lng;updateMap();updateCoords()});
    document.querySelectorAll('.stype').forEach(el=>{el.onclick=()=>{el.classList.toggle('active');document.querySelectorAll('.preset').forEach(p=>p.classList.remove('active'))}});
    init3D();
}
function init3D(){
    const container=$('preview3d');scene=new THREE.Scene();scene.background=new THREE.Color(0x1e1e1e);
    camera=new THREE.PerspectiveCamera(45,container.clientWidth/container.clientHeight,0.1,1000);camera.position.set(80,80,80);camera.lookAt(0,0,0);
    renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(container.clientWidth,container.clientHeight);
    scene.add(new THREE.AmbientLight(0xffffff,0.6));const dir=new THREE.DirectionalLight(0xffffff,0.8);dir.position.set(50,100,50);scene.add(dir);
    const grid=new THREE.GridHelper(100,10,0x333333,0x222222);grid.rotation.x=Math.PI/2;scene.add(grid);
}
function updateMap(){const r=+$('rad').value;if(mapMarker)map.removeLayer(mapMarker);if(circle)map.removeLayer(circle);mapMarker=L.marker([lat,lon],{icon:L.divIcon({html:'<div style="width:12px;height:12px;background:#00AE42;border:2px solid #fff;border-radius:50%"></div>',iconSize:[12,12],iconAnchor:[6,6]})}).addTo(map);circle=L.circle([lat,lon],{radius:r,color:'#00AE42',fillOpacity:.05,weight:2}).addTo(map)}
function updRad(){const r=$('rad').value;$('radV').textContent=r>=1000?(r/1000)+'km':r+'m';if(circle)circle.setRadius(r)}
function updateCoords(){$('latV').textContent=lat.toFixed(5);$('lonV').textContent=lon.toFixed(5)}
function setMode(m){mode=m;document.querySelectorAll('.mode').forEach(el=>el.classList.toggle('active',el.dataset.mode===m));$('statMode').textContent=m.charAt(0).toUpperCase()+m.slice(1);document.querySelectorAll('.street-sec').forEach(el=>el.style.display=m==='streets'?'block':'none')}
function preset(name){const types=presets[name]||[];document.querySelectorAll('.stype').forEach(el=>el.classList.toggle('active',types.includes(el.dataset.t)));document.querySelectorAll('.preset').forEach(p=>p.classList.toggle('active',p.textContent.toLowerCase().includes(name)))}
function getSelectedTypes(){return Array.from(document.querySelectorAll('.stype.active')).map(el=>el.dataset.t)}
function setMarker(m){markerType=m;document.querySelectorAll('.marker').forEach(el=>el.classList.toggle('active',el.dataset.m===m));$('markerOpts').style.display=m==='none'?'none':'block'}
async function search(){const q=$('loc').value.trim();if(!q)return;try{const r=await fetch('/api/geocode?q='+encodeURIComponent(q));const d=await r.json();if(d.error)throw new Error(d.error);lat=d.lat;lon=d.lon;map.setView([lat,lon],14);updateMap();updateCoords();msg('success','✓ '+d.name.substring(0,30))}catch(e){msg('error',e.message)}}
async function preview(){const btn=$('btnPreview');btn.disabled=true;btn.innerHTML='<div class="spinner"></div>';try{const r=await fetch('/api/map/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon,radius:+$('rad').value,size:+$('size').value,height:+$('ht').value,lineWidth:+$('lw').value,mode,streetTypes:getSelectedTypes(),marker:markerType,markerSize:+$('ms').value,markerGap:+$('mg').value})});const d=await r.json();if(d.error)throw new Error(d.error);load3DPreview(d.vertices,d.faces);$('statData').textContent=d.count+(mode==='city'?' bldgs':' segs');$('badge').style.display='inline';msg('success','✓ Preview loaded')}catch(e){msg('error',e.message)}finally{btn.disabled=false;btn.innerHTML='👁 Preview'}}
function load3DPreview(verts,faces){if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose()}const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));geom.setIndex(faces);geom.computeVertexNormals();const mat=new THREE.MeshPhongMaterial({color:0x00AE42,flatShading:true});mesh=new THREE.Mesh(geom,mat);geom.computeBoundingBox();const box=geom.boundingBox;const center=new THREE.Vector3();box.getCenter(center);mesh.position.sub(center);const maxDim=Math.max(box.max.x-box.min.x,box.max.y-box.min.y,box.max.z-box.min.z);const scale=60/maxDim;mesh.scale.set(scale,scale,scale);scene.add(mesh);const container=$('preview3d');container.innerHTML='';container.appendChild(renderer.domElement);let angle=0;function animate(){requestAnimationFrame(animate);angle+=0.005;camera.position.x=Math.sin(angle)*100;camera.position.z=Math.cos(angle)*100;camera.lookAt(0,0,0);renderer.render(scene,camera)}animate()}
async function exportSTL(){const btn=$('btnExport');btn.disabled=true;btn.innerHTML='<div class="spinner"></div>';try{const r=await fetch('/api/map/stl',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon,radius:+$('rad').value,size:+$('size').value,height:+$('ht').value,lineWidth:+$('lw').value,mode,streetTypes:getSelectedTypes(),marker:markerType,markerSize:+$('ms').value,markerGap:+$('mg').value})});if(!r.ok)throw new Error((await r.json()).error);const blob=await r.blob();const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`map_${lat.toFixed(4)}_${lon.toFixed(4)}_${mode}.stl`;a.click();msg('success','✓ STL downloaded!')}catch(e){msg('error',e.message)}finally{btn.disabled=false;btn.innerHTML='⬇ Export STL'}}
function msg(type,text){const el=$('status');el.className='status '+type;el.textContent=text;setTimeout(()=>el.className='status',4000)}
document.addEventListener('DOMContentLoaded',init);
</script>
</body></html>'''

# ==================== ROUTES ====================
@app.route('/')
def landing():return render_template_string(LANDING_HTML)

@app.route('/map')
def map_tool():return render_template_string(MAP_HTML)

@app.route('/image')
def image_tool():return render_template_string(IMAGE_HTML)

@app.route('/api/geocode')
def api_geo():
    q=request.args.get('q','')
    r=geocode(q) if q else None
    return jsonify(r) if r else (jsonify({'error':'Not found'}),404)

@app.route('/api/map/preview',methods=['POST'])
def api_map_preview():
    try:
        d=request.json
        lat,lon,radius,size,height=d['lat'],d['lon'],d['radius'],d['size'],d['height']
        mode=d['mode'];lw=d.get('lineWidth',1.2)
        street_types=d.get('streetTypes',['motorway','trunk','primary','secondary','tertiary','residential'])
        marker=d.get('marker','none');marker_size=d.get('markerSize',12);marker_gap=d.get('markerGap',2)
        if mode=='streets':
            gdf=fetch_streets(lat,lon,radius,street_types)
            mesh=create_streets_mesh(gdf,lat,lon,radius,size,height,lw,marker,marker_size,marker_gap)
            count=len(gdf)
        elif mode=='city':
            mesh=create_city_mesh(lat,lon,radius,size,height,lw,1.0,1.5)
            count=len(fetch_buildings(lat,lon,radius))
        else:
            mesh=create_terrain_mesh(lat,lon,radius,size,height*5,1.5)
            count=80*80
        return jsonify({'vertices':mesh.vertices.flatten().tolist(),'faces':mesh.faces.flatten().tolist(),'count':count})
    except Exception as e:return jsonify({'error':str(e)}),500

@app.route('/api/map/stl',methods=['POST'])
def api_map_stl():
    try:
        d=request.json
        lat,lon,radius,size,height=d['lat'],d['lon'],d['radius'],d['size'],d['height']
        mode=d['mode'];lw=d.get('lineWidth',1.2)
        street_types=d.get('streetTypes',['motorway','trunk','primary','secondary','tertiary','residential'])
        marker=d.get('marker','none');marker_size=d.get('markerSize',12);marker_gap=d.get('markerGap',2)
        if mode=='streets':
            gdf=fetch_streets(lat,lon,radius,street_types)
            mesh=create_streets_mesh(gdf,lat,lon,radius,size,height,lw,marker,marker_size,marker_gap)
        elif mode=='city':
            mesh=create_city_mesh(lat,lon,radius,size,height,lw,1.0,1.5)
        else:
            mesh=create_terrain_mesh(lat,lon,radius,size,height*5,1.5)
        mesh.fill_holes();mesh.fix_normals()
        buf=io.BytesIO();mesh.export(buf,file_type='stl');buf.seek(0)
        return send_file(buf,mimetype='application/octet-stream',as_attachment=True,download_name='map.stl')
    except Exception as e:return jsonify({'error':str(e)}),500

@app.route('/api/image/convert',methods=['POST'])
def api_image_convert():
    try:
        d=request.json
        img_b64=d['image']
        # Remove data URL prefix if present
        if ',' in img_b64:
            img_b64=img_b64.split(',')[1]
        img_data=base64.b64decode(img_b64)
        
        mode=d.get('mode','outline')
        threshold=d.get('threshold',128)
        blur=d.get('blur',1)
        simplify=d.get('simplify',2)
        invert=d.get('invert',False)
        
        if mode=='filled':
            svg,paths,w,h=image_to_filled_svg(img_data,threshold,blur,simplify,invert)
        else:
            svg,paths,w,h=image_to_svg(img_data,threshold,blur,simplify,invert,mode)
        
        return jsonify({'svg':svg,'paths':paths,'width':w,'height':h})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/health')
def health():return jsonify({'status':'ok','version':'1.0','tools':['map','image'],'author':'SebGE'})

if __name__=='__main__':
    port=int(os.environ.get('PORT',8080))
    print(f"\n  🛠️  SebGE Tools v1.0\n  → http://localhost:{port}\n  Tools: Map→STL, Image→SVG\n")
    app.run(host='0.0.0.0',port=port,debug=os.environ.get('DEBUG','false').lower()=='true')
