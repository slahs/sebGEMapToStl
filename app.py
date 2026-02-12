# -*- coding: utf-8 -*-
"""SebGE Tools v1.2 - Map/Image/Print to STL"""
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
    
    # Try multiple Overpass servers
    servers = [
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
        'https://maps.mail.ru/osm/tools/overpass/api/interpreter'
    ]
    
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
                        if len(coords)>=2:
                            lines.append(LineString(coords))
                if lines:
                    print(f"Fetched {len(lines)} streets from {server}")
                    return gpd.GeoDataFrame(geometry=lines,crs='EPSG:4326')
        except Exception as ex:
            print(f"Server {server} failed: {ex}")
            continue
    
    # If all servers fail, raise error instead of returning fake data
    raise ValueError(f"Could not fetch street data for {lat},{lon}. All Overpass servers failed.")

def fetch_buildings(lat,lon,radius):
    bbox=radius/111000;q=f'[out:json][timeout:90];way["building"]({lat-bbox},{lon-bbox*1.5},{lat+bbox},{lon+bbox*1.5});out body;>;out skel qt;'
    
    servers = [
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
        'https://maps.mail.ru/osm/tools/overpass/api/interpreter'
    ]
    
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
            if buildings:
                print(f"Fetched {len(buildings)} buildings from {server}")
                return buildings
        except Exception as ex:
            print(f"Buildings server {server} failed: {ex}")
            continue
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

# === IMAGE TO SVG ===
def smooth_contour(contour, sigma):
    if sigma <= 0 or len(contour) < 5:return contour
    smoothed = np.zeros_like(contour)
    smoothed[:, 0] = gaussian_filter1d(contour[:, 0], sigma=sigma, mode='wrap')
    smoothed[:, 1] = gaussian_filter1d(contour[:, 1], sigma=sigma, mode='wrap')
    return smoothed

def image_to_svg(img_data, threshold=128, blur=1, simplify=2, smooth=0, invert=False, mode='outline', single_line=False):
    img = Image.open(io.BytesIO(img_data))
    if img.mode == 'RGBA':bg = Image.new('RGB', img.size, (255, 255, 255));bg.paste(img, mask=img.split()[3]);img = bg
    elif img.mode != 'RGB':img = img.convert('RGB')
    w, h = img.size;gray = img.convert('L')
    if blur > 0:gray = gray.filter(ImageFilter.GaussianBlur(radius=blur))
    if invert:gray = ImageOps.invert(gray)
    img_array = np.array(gray)
    if single_line:
        binary = img_array < threshold;skeleton = skeletonize(binary)
        points = np.argwhere(skeleton);paths = []
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/>'
        if len(points) > 0:
            svg += f'<path d="M {points[0][1]} {points[0][0]}'
            for p in points[1::max(1,simplify)]:svg += f' L {p[1]} {p[0]}'
            svg += '" fill="none" stroke="black" stroke-width="1"/>'
        svg += '</svg>';return svg, len(points)//max(1,simplify), w, h
    if mode == 'outline':
        edges = feature.canny(img_array, sigma=blur+1, low_threshold=threshold*0.3, high_threshold=threshold)
        binary = edges.astype(np.uint8) * 255
    else:binary = (img_array < threshold).astype(np.uint8) * 255
    contours = measure.find_contours(binary, 0.5)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/>'
    paths = []
    for contour in contours:
        if len(contour) < 3:continue
        if smooth > 0:contour = smooth_contour(contour, sigma=smooth)
        simplified = contour[::max(1, simplify)]
        if len(simplified) < 3:continue
        d = f"M {simplified[0][1]:.1f},{simplified[0][0]:.1f}"
        for p in simplified[1:]:d += f" L {p[1]:.1f},{p[0]:.1f}"
        d += " Z";paths.append(d)
    if paths:svg += f'<path d="{" ".join(paths)}" fill="none" stroke="black" stroke-width="1"/>'
    svg += '</svg>';return svg, len(paths), w, h

def image_to_filled_svg(img_data, threshold=128, blur=1, simplify=2, smooth=0, invert=False):
    img = Image.open(io.BytesIO(img_data))
    if img.mode == 'RGBA':bg = Image.new('RGB', img.size, (255, 255, 255));bg.paste(img, mask=img.split()[3]);img = bg
    elif img.mode != 'RGB':img = img.convert('RGB')
    w, h = img.size;gray = img.convert('L')
    if blur > 0:gray = gray.filter(ImageFilter.GaussianBlur(radius=blur))
    if invert:gray = ImageOps.invert(gray)
    img_array = np.array(gray);binary = (img_array < threshold).astype(np.uint8) * 255
    contours = measure.find_contours(binary, 0.5)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/>'
    paths = []
    for contour in contours:
        if len(contour) < 3:continue
        if smooth > 0:contour = smooth_contour(contour, sigma=smooth)
        simplified = contour[::max(1, simplify)]
        if len(simplified) < 3:continue
        d = f"M {simplified[0][1]:.1f},{simplified[0][0]:.1f}"
        for p in simplified[1:]:d += f" L {p[1]:.1f},{p[0]:.1f}"
        d += " Z";paths.append(d)
    if paths:svg += f'<path d="{" ".join(paths)}" fill="black" stroke="none" fill-rule="evenodd"/>'
    svg += '</svg>';return svg, len(paths), w, h

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
    # Denoise
    if denoise > 0:gray = gaussian(gray, sigma=denoise, preserve_range=True)
    # Detect background (assume corners are background)
    h, w = gray.shape
    corner_size = max(10, min(h, w) // 10)
    corners = [gray[:corner_size, :corner_size], gray[:corner_size, -corner_size:], 
               gray[-corner_size:, :corner_size], gray[-corner_size:, -corner_size:]]
    bg_value = np.median([np.median(c) for c in corners])
    # Normalize: background becomes white (255), darker areas become darker
    if bg_value > 128:  # Light background (normal case)
        # Stretch contrast: bg_value->255 (white), darkest->0 (black)
        p_dark = np.percentile(gray, 2)
        gray = np.clip((gray - p_dark) / (bg_value - p_dark + 1e-8) * 255, 0, 255)
    else:  # Dark background - invert
        p_light = np.percentile(gray, 98)
        gray = 255 - np.clip((gray - bg_value) / (p_light - bg_value + 1e-8) * 255, 0, 255)
    # Apply contrast enhancement
    if enhance_contrast > 1:
        mid = 128
        gray = mid + (gray - mid) * enhance_contrast
        gray = np.clip(gray, 0, 255)
    # Final normalization to use full range
    p2, p98 = np.percentile(gray, (1, 99))
    gray = exposure.rescale_intensity(gray, in_range=(p2, p98), out_range=(0, 255))
    return gray.astype(np.uint8)

def create_hueforge_preview(gray, num_colors=2):
    h, w = gray.shape;preview = np.zeros((h, w, 3), dtype=np.uint8)
    # HueForge: White (255) = thin = light areas, Black (0) = thick = dark areas
    if num_colors == 2:
        # 2 colors: White base, Black on top for dark areas
        preview[gray >= 128] = [255, 255, 255]  # Light areas stay white
        preview[gray < 128] = [30, 30, 30]       # Dark areas become black
    else:
        # 3 colors: White base, Gray middle, Black for darkest
        preview[gray >= 170] = [255, 255, 255]  # Lightest = white
        preview[(gray >= 85) & (gray < 170)] = [120, 120, 120]  # Mid = gray
        preview[gray < 85] = [30, 30, 30]        # Darkest = black
    return preview

def create_relief_stl(gray, width_mm=100, total_height_mm=2.4, base_height_mm=0.6, border_mm=2):
    h, w = gray.shape;aspect = h / w;width = width_mm;height = width * aspect
    if border_mm > 0:
        padded = np.pad(gray, pad_width=int(border_mm * w / width_mm), mode='constant', constant_values=255)
        gray = padded;h, w = gray.shape;height = width * (h / w)
    max_res = 400
    if w > max_res or h > max_res:
        scale = max_res / max(w, h);new_w, new_h = int(w * scale), int(h * scale)
        img = Image.fromarray(gray);img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        gray = np.array(img);h, w = gray.shape
    relief_height = total_height_mm - base_height_mm
    normalized = 1.0 - (gray.astype(float) / 255.0)
    heights = base_height_mm + normalized * relief_height
    step_x = width / (w - 1);step_y = height / (h - 1)
    vertices = [];faces = []
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
    for j in range(w - 1):
        t0, t1 = (h-1)*n+j, (h-1)*n+j+1;b0, b1 = off+(h-1)*n+j, off+(h-1)*n+j+1
        faces.append([t0, b0, t1]);faces.append([t1, b0, b1])
    for i in range(h - 1):
        t0, t1 = i*n, (i+1)*n;b0, b1 = off+i*n, off+(i+1)*n
        faces.append([t0, b0, t1]);faces.append([t1, b0, b1])
    for i in range(h - 1):
        t0, t1 = i*n+w-1, (i+1)*n+w-1;b0, b1 = off+i*n+w-1, off+(i+1)*n+w-1
        faces.append([t0, t1, b0]);faces.append([t1, b1, b0])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces);mesh.fix_normals()
    return mesh, width, height

def calculate_print_instructions(total_height_mm, base_height_mm, layer_height_mm, num_colors):
    total_layers = int(round(total_height_mm / layer_height_mm))
    base_layers = int(round(base_height_mm / layer_height_mm))
    if num_colors == 2:
        return {'total_layers': total_layers, 'summary': f"Layer Height: {layer_height_mm}mm | Total: {total_height_mm}mm | Layers: {total_layers}\n\n1. Start WHITE\n2. Change BLACK at Layer {base_layers} (Z={base_height_mm}mm)"}
    else:
        relief = total_height_mm - base_height_mm
        g_layer = int(round((base_height_mm + relief * 0.33) / layer_height_mm))
        b_layer = int(round((base_height_mm + relief * 0.66) / layer_height_mm))
        return {'total_layers': total_layers, 'summary': f"Layer Height: {layer_height_mm}mm | Total: {total_height_mm}mm | Layers: {total_layers}\n\n1. Start WHITE\n2. GRAY at Layer {g_layer}\n3. BLACK at Layer {b_layer}"}

# === HTML TEMPLATES ===

# === HTML TEMPLATES ===
LANDING = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SebGE Tools - 3D Print & Design Tools</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#fff;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 20px}
.header{text-align:center;margin-bottom:40px}
.logo-box{display:inline-flex;align-items:center;gap:16px;margin-bottom:16px}
.logo-icon{width:72px;height:72px;background:linear-gradient(135deg,#00AE42,#008F36);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:36px;box-shadow:0 0 30px rgba(0,174,66,0.3)}
.logo-text h1{font-size:36px;font-weight:700;letter-spacing:-1px}
.logo-text .sub{color:#00AE42;font-size:12px;font-weight:600;letter-spacing:2px;margin-top:4px}
.creator{display:inline-flex;align-items:center;gap:12px;background:rgba(0,174,66,0.1);border:1px solid rgba(0,174,66,0.3);border-radius:50px;padding:8px 20px 8px 8px;margin-top:16px;text-decoration:none;color:#fff;transition:all .2s}
.creator:hover{background:rgba(0,174,66,0.2);border-color:#00AE42}
.creator-avatar{width:36px;height:36px;background:#00AE42;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px}
.creator-info{text-align:left}
.creator-name{font-size:13px;font-weight:600;color:#00AE42}
.creator-link{font-size:11px;color:#888}
.tools{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;max-width:900px;width:100%;margin-bottom:40px}
.tool{background:#151515;border:1px solid #2a2a2a;border-radius:16px;padding:32px 24px;text-decoration:none;color:#fff;transition:all .2s;position:relative;overflow:hidden}
.tool:hover{border-color:#00AE42;transform:translateY(-4px);box-shadow:0 10px 40px rgba(0,0,0,0.3)}
.tool-badge{position:absolute;top:16px;right:16px;background:#00AE42;color:#fff;font-size:9px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.5px}
.tool-badge.vector{background:#9333EA}
.tool-icon{font-size:48px;margin-bottom:16px;display:block}
.tool h3{font-size:20px;font-weight:700;margin-bottom:8px}
.tool p{font-size:13px;color:#888;line-height:1.5}
.support{background:linear-gradient(135deg,#1a1a1a,#151515);border:1px solid #2a2a2a;border-radius:16px;padding:24px 32px;max-width:900px;width:100%;display:flex;align-items:center;gap:24px;margin-bottom:30px}
.support-icon{font-size:40px;flex-shrink:0}
.support-text{flex:1}
.support-text h4{font-size:15px;font-weight:600;margin-bottom:6px;color:#fff}
.support-text p{font-size:12px;color:#888;line-height:1.5}
.support-btn{background:#00AE42;color:#fff;text-decoration:none;padding:12px 24px;border-radius:8px;font-size:13px;font-weight:600;white-space:nowrap;transition:all .2s}
.support-btn:hover{background:#00C94B;transform:scale(1.05)}
.support-price{text-align:center;margin-right:8px}
.support-price .amount{font-size:24px;font-weight:700;color:#00AE42}
.support-price .period{font-size:10px;color:#888}
.footer{text-align:center;font-size:12px;color:#666;margin-top:auto;padding-top:20px}
.footer a{color:#00AE42;text-decoration:none}
.footer span{margin:0 8px}
@media(max-width:700px){
    .logo-box{flex-direction:column;gap:12px}
    .logo-text{text-align:center}
    .logo-text h1{font-size:28px}
    .tools{grid-template-columns:1fr}
    .support{flex-direction:column;text-align:center;gap:16px}
    .support-price{margin:0}
}
</style></head><body>
<div class="header">
<div class="logo-box">
<div class="logo-icon">🛠️</div>
<div class="logo-text">
<h1>SebGE Tools</h1>
<div class="sub">3D PRINT & DESIGN TOOLS</div>
</div>
</div>
<a href="https://makerworld.com/de/@SebGE" target="_blank" class="creator">
<div class="creator-avatar">S</div>
<div class="creator-info">
<div class="creator-name">Created by SebGE</div>
<div class="creator-link">MakerWorld Profile →</div>
</div>
</a>
</div>

<div class="tools">
<a href="/map" class="tool">
<span class="tool-badge">3D PRINT</span>
<span class="tool-icon">🗺️</span>
<h3>Map → STL</h3>
<p>Create 3D printable maps from any location. Streets, cities, or terrain with real elevation data.</p>
</a>
<a href="/image" class="tool">
<span class="tool-badge vector">VECTOR</span>
<span class="tool-icon">🎨</span>
<h3>Image → SVG</h3>
<p>Convert images to vector graphics. Perfect for laser cutting, vinyl cutting, or clean scaling.</p>
</a>
</div>

<div class="support">
<div class="support-icon">❤️</div>
<div class="support-text">
<h4>Support this Project</h4>
<p>These tools are free to use! A $3/month supporter membership on MakerWorld helps cover filament for new ideas and server costs to keep this tool running.</p>
</div>
<div class="support-price">
<div class="amount">$3</div>
<div class="period">/month</div>
</div>
<a href="https://makerworld.com/de/@SebGE" target="_blank" class="support-btn">Support on MakerWorld</a>
</div>

<div class="footer">
Made with ❤️ by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a>
<span>•</span>
Open Source Tools for Makers
</div>
</body></html>"""

IMAGE_HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Image to SVG | SebGE Tools</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--g:#00AE42;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}
body{font-family:system-ui;background:var(--bg);color:var(--t);min-height:100vh}
.app{display:grid;grid-template-columns:260px 1fr 200px;height:100vh}
.panel{background:var(--c);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.back{color:var(--t2);text-decoration:none;font-size:10px}
.back:hover{color:var(--g)}
.logo{display:flex;align-items:center;gap:8px;padding-bottom:8px;border-bottom:1px solid var(--br)}
.logo-icon{width:32px;height:32px;background:var(--g);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px}
.logo h1{font-size:13px}
.logo small{font-size:7px;color:var(--g)}
.sec{background:var(--c2);border-radius:6px;padding:10px}
.sec-title{font-size:8px;color:var(--g);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-weight:600}
.upload{border:2px dashed var(--br);border-radius:6px;padding:20px 10px;text-align:center;cursor:pointer}
.upload:hover{border-color:var(--g)}
.upload p{font-size:10px;color:var(--t2)}
.upload.has-img{padding:8px}
.upload img{max-width:100%;max-height:100px;border-radius:4px}
#file{display:none}
.modes{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.mode{padding:8px 4px;background:var(--c);border:2px solid var(--br);border-radius:4px;cursor:pointer;text-align:center;font-size:9px}
.mode:hover,.mode.active{border-color:var(--g)}
.slider{margin-bottom:4px}
.slider-head{display:flex;justify-content:space-between;margin-bottom:2px}
.slider label{font-size:9px;color:var(--t2)}
.slider .val{font-size:9px;color:var(--g);font-family:monospace}
.slider input{width:100%;height:4px;background:var(--c);border-radius:2px;-webkit-appearance:none}
.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--g);border-radius:50%;cursor:pointer}
.check{display:flex;align-items:center;gap:6px;font-size:10px;cursor:pointer;margin-top:6px}
.check input{width:14px;height:14px;accent-color:var(--g)}
.btn{padding:10px;border:none;border-radius:6px;font-size:10px;font-weight:600;cursor:pointer;width:100%;margin-bottom:4px}
.btn-primary{background:var(--g);color:#fff}
.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.preview-panel{background:#fff;display:flex;align-items:center;justify-content:center;padding:20px}
.preview-panel img,.preview-panel svg{max-width:100%;max-height:80vh}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.stat{background:var(--c2);border-radius:4px;padding:6px;text-align:center}
.stat label{font-size:7px;color:var(--t2);text-transform:uppercase}
.stat .v{font-size:11px;color:var(--g);font-family:monospace}
.status{padding:6px;border-radius:4px;font-size:9px;display:none;text-align:center}
.status.error{display:block;background:rgba(239,68,68,.1);color:#ef4444}
.status.success{display:block;background:rgba(0,174,66,.1);color:var(--g)}
.footer{margin-top:auto;text-align:center;font-size:8px;color:var(--t2);padding-top:8px;border-top:1px solid var(--br)}
.footer a{color:var(--g)}
.mobile-tabs{display:none;position:fixed;bottom:0;left:0;right:0;background:var(--c);border-top:1px solid var(--br);padding:8px;z-index:100}
.mobile-tabs .tabs{display:flex;gap:4px}
.mobile-tabs .tab{flex:1;padding:10px;text-align:center;background:var(--c2);border-radius:6px;font-size:10px;cursor:pointer}
.mobile-tabs .tab.active{background:var(--g)}
@media(max-width:800px){
    .app{display:block;height:auto;padding-bottom:70px}
    .panel,.preview-panel{display:none;min-height:calc(100vh - 70px)}
    .panel.active,.preview-panel.active{display:flex}
    .preview-panel.active{display:flex}
    .mobile-tabs{display:block}
}
</style></head><body>
<div class="app">
<div class="panel active" id="panelSettings">
<a href="/" class="back">← Back</a>
<div class="logo"><div class="logo-icon">I</div><div><h1>Image → SVG</h1><small>VECTORIZE IMAGES</small></div></div>
<div class="upload" id="up" onclick="document.getElementById('file').click()"><p>Drop image here<br>or tap to select</p></div>
<input type="file" id="file" accept="image/*">
<div class="sec">
<div class="sec-title">Mode</div>
<div class="modes">
<div class="mode active" data-m="outline" onclick="setMode('outline')">Outline</div>
<div class="mode" data-m="filled" onclick="setMode('filled')">Filled</div>
<div class="mode" data-m="threshold" onclick="setMode('threshold')">Threshold</div>
<div class="mode" data-m="centerline" onclick="setMode('centerline')">Single Line</div>
</div>
</div>
<div class="sec">
<div class="sec-title">Settings</div>
<div class="slider"><div class="slider-head"><label>Threshold</label><span class="val" id="thV">128</span></div><input type="range" id="th" min="10" max="245" value="128" oninput="$('thV').textContent=this.value"></div>
<div class="slider"><div class="slider-head"><label>Blur</label><span class="val" id="blV">1</span></div><input type="range" id="bl" min="0" max="5" value="1" oninput="$('blV').textContent=this.value"></div>
<div class="slider"><div class="slider-head"><label>Simplify</label><span class="val" id="siV">2</span></div><input type="range" id="si" min="1" max="10" value="2" oninput="$('siV').textContent=this.value"></div>
<div class="slider"><div class="slider-head"><label>Smooth</label><span class="val" id="smV">0</span></div><input type="range" id="sm" min="0" max="10" value="0" oninput="$('smV').textContent=this.value"></div>
<label class="check"><input type="checkbox" id="inv"> Invert Colors</label>
</div>
<button class="btn btn-secondary" id="btnConvert" onclick="convert()" disabled>Convert</button>
<button class="btn btn-primary" id="btnExport" onclick="exportSVG()" disabled>⬇ Download SVG</button>
<div class="status" id="status"></div>
<div class="footer">Made by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div>
</div>
<div class="preview-panel" id="panelPreview"><p style="color:#888">Upload an image</p></div>
<div class="panel" id="panelInfo">
<div class="sec-title">Output Info</div>
<div class="stats">
<div class="stat"><label>Paths</label><div class="v" id="stP">-</div></div>
<div class="stat"><label>Size</label><div class="v" id="stS">-</div></div>
</div>
</div>
</div>
<div class="mobile-tabs">
<div class="tabs">
<div class="tab active" onclick="showPanel('Settings')">Settings</div>
<div class="tab" onclick="showPanel('Preview')">Preview</div>
<div class="tab" onclick="showPanel('Info')">Info</div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
let mode='outline',imgData=null,svgRes=null;
const up=$('up');
up.ondragover=e=>{e.preventDefault();up.style.borderColor='#00AE42'};
up.ondragleave=()=>up.style.borderColor='#2a2a2a';
up.ondrop=e=>{e.preventDefault();up.style.borderColor='#2a2a2a';if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])};
$('file').onchange=e=>{if(e.target.files.length)handleFile(e.target.files[0])};

function handleFile(f){
    if(!f.type.startsWith('image/')){msg('error','Please upload an image');return}
    const reader=new FileReader();
    reader.onload=e=>{
        imgData=e.target.result;
        up.innerHTML='<img src="'+imgData+'">';
        up.classList.add('has-img');
        $('btnConvert').disabled=false;
        $('panelPreview').innerHTML='<img src="'+imgData+'">';
        msg('success','Image loaded');
    };
    reader.readAsDataURL(f);
}

function setMode(m){
    mode=m;
    document.querySelectorAll('.mode').forEach(e=>e.classList.toggle('active',e.dataset.m===m));
}

async function convert(){
    if(!imgData)return;
    $('btnConvert').disabled=true;
    $('btnConvert').textContent='Converting...';
    try{
        const r=await fetch('/api/image/convert',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                image:imgData,
                mode,
                threshold:parseInt($('th').value),
                blur:parseInt($('bl').value),
                simplify:parseInt($('si').value),
                smooth:parseInt($('sm').value),
                invert:$('inv').checked,
                singleLine:mode==='centerline'
            })
        });
        const d=await r.json();
        if(d.error)throw new Error(d.error);
        svgRes=d.svg;
        $('panelPreview').innerHTML=svgRes;
        $('stP').textContent=d.paths;
        $('stS').textContent=d.width+'x'+d.height;
        $('btnExport').disabled=false;
        msg('success','Converted!');
    }catch(e){
        msg('error',e.message);
    }finally{
        $('btnConvert').disabled=false;
        $('btnConvert').textContent='Convert';
    }
}

function exportSVG(){
    if(!svgRes)return;
    const blob=new Blob([svgRes],{type:'image/svg+xml'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='vectorized.svg';
    a.click();
    msg('success','SVG downloaded!');
}

function showPanel(name){
    document.querySelectorAll('.panel,.preview-panel').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.mobile-tabs .tab').forEach(t=>t.classList.remove('active'));
    $('panel'+name).classList.add('active');
    event.target.classList.add('active');
}

function msg(type,text){
    const el=$('status');
    el.className='status '+type;
    el.textContent=text;
    setTimeout(()=>el.className='status',3000);
}
</script>
</body></html>"""

PRINT_HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Print to STL | SebGE Tools</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--g:#00AE42;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}
body{font-family:system-ui;background:var(--bg);color:var(--t);min-height:100vh}
.app{display:grid;grid-template-columns:300px 1fr 260px;height:100vh}
.panel{background:var(--c);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.panel::-webkit-scrollbar{width:4px}
.panel::-webkit-scrollbar-thumb{background:var(--br);border-radius:2px}
.back{color:var(--t2);text-decoration:none;font-size:10px}
.back:hover{color:var(--g)}
.logo{display:flex;align-items:center;gap:8px;padding-bottom:8px;border-bottom:1px solid var(--br)}
.logo-icon{width:32px;height:32px;background:var(--g);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px}
.logo h1{font-size:13px}
.logo small{font-size:7px;color:var(--g)}
.sec{background:var(--c2);border-radius:6px;padding:10px}
.sec-title{font-size:8px;color:var(--g);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-weight:600}
.upload{border:2px dashed var(--br);border-radius:6px;padding:20px 10px;text-align:center;cursor:pointer}
.upload:hover{border-color:var(--g)}
.upload p{font-size:10px;color:var(--t2)}
.upload.has-img{padding:8px}
.upload img{max-width:100%;max-height:100px;border-radius:4px}
#file{display:none}
.color-count{display:flex;gap:4px}
.cc-btn{flex:1;padding:8px 4px;background:var(--c);border:2px solid var(--br);border-radius:4px;cursor:pointer;text-align:center;font-size:11px;font-weight:600}
.cc-btn:hover,.cc-btn.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.layer-row{display:flex;align-items:center;gap:8px;padding:8px;background:var(--c);border-radius:6px;margin-bottom:6px}
.layer-row .color-pick{width:36px;height:36px;border-radius:6px;border:2px solid var(--br);cursor:pointer;padding:0;overflow:hidden}
.layer-row .color-pick::-webkit-color-swatch-wrapper{padding:0}
.layer-row .color-pick::-webkit-color-swatch{border:none}
.layer-row .layer-info{flex:1}
.layer-row .layer-name{font-size:10px;font-weight:600;margin-bottom:2px}
.layer-row .layer-desc{font-size:8px;color:var(--t2)}
.layer-row input[type=range]{width:100%;height:4px;background:var(--c2);border-radius:2px;-webkit-appearance:none;margin-top:4px}
.layer-row input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;background:var(--g);border-radius:50%;cursor:pointer}
.slider{margin-bottom:4px}
.slider-head{display:flex;justify-content:space-between;margin-bottom:2px}
.slider label{font-size:9px;color:var(--t2)}
.slider .val{font-size:9px;color:var(--g);font-family:monospace}
.slider input[type=range]{width:100%;height:4px;background:var(--c);border-radius:2px;-webkit-appearance:none}
.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:var(--g);border-radius:50%;cursor:pointer}
.btn{padding:10px;border:none;border-radius:6px;font-size:10px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:4px;width:100%}
.btn-primary{background:var(--g);color:#fff}
.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.preview-panel{background:var(--c);display:flex;flex-direction:column}
.preview-header{padding:12px;border-bottom:1px solid var(--br);display:flex;justify-content:space-between;align-items:center}
.preview-header h2{font-size:12px}
.preview-header .badge{font-size:8px;background:var(--g);padding:2px 6px;border-radius:8px}
.preview-area{flex:1;display:flex;align-items:center;justify-content:center;padding:20px;background:#111}
.preview-area canvas{max-width:100%;max-height:100%;border-radius:6px}
.instr{background:var(--c2);font-family:monospace;font-size:9px;padding:10px;border-radius:6px;white-space:pre-wrap;max-height:140px;overflow-y:auto;line-height:1.5}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:6px}
.stat{background:var(--c2);border-radius:4px;padding:6px;text-align:center}
.stat label{font-size:7px;color:var(--t2);text-transform:uppercase}
.stat .v{font-size:11px;color:var(--g);font-family:monospace;font-weight:600}
.status{padding:6px;border-radius:4px;font-size:9px;display:none;text-align:center}
.status.error{display:block;background:rgba(239,68,68,.1);color:#ef4444}
.status.success{display:block;background:rgba(0,174,66,.1);color:var(--g)}
.footer{margin-top:auto;text-align:center;font-size:8px;color:var(--t2);padding-top:8px;border-top:1px solid var(--br)}
.footer a{color:var(--g)}
.mobile-tabs{display:none;position:fixed;bottom:0;left:0;right:0;background:var(--c);border-top:1px solid var(--br);padding:8px;z-index:100}
.mobile-tabs .tabs{display:flex;gap:4px}
.mobile-tabs .tab{flex:1;padding:10px;text-align:center;background:var(--c2);border-radius:6px;font-size:10px;cursor:pointer}
.mobile-tabs .tab.active{background:var(--g)}
@media(max-width:900px){
    .app{display:block;height:auto;padding-bottom:70px}
    .panel{display:none;min-height:calc(100vh - 70px)}
    .panel.active{display:flex}
    .preview-panel{display:none}
    .preview-panel.active{display:flex;min-height:calc(100vh - 70px)}
    .mobile-tabs{display:block}
}
</style></head><body>
<div class="app">
<div class="panel active" id="panelSettings">
<a href="/" class="back">← Back</a>
<div class="logo"><div class="logo-icon">P</div><div><h1>Print → STL</h1><small>HUEFORGE STYLE</small></div></div>
<div class="upload" id="up" onclick="document.getElementById('file').click()"><p>Drop image here<br>or tap to select</p></div>
<input type="file" id="file" accept="image/*">

<div class="sec">
<div class="sec-title">Number of Colors</div>
<div class="color-count">
<div class="cc-btn active" onclick="setColorCount(2)">2</div>
<div class="cc-btn" onclick="setColorCount(3)">3</div>
<div class="cc-btn" onclick="setColorCount(4)">4</div>
</div>
</div>

<div class="sec" id="layersSec">
<div class="sec-title">Layer Colors & Thresholds</div>
<div id="layerControls"></div>
</div>

<div class="sec">
<div class="sec-title">Model Size</div>
<div class="slider"><div class="slider-head"><label>Width</label><span class="val" id="wV">100mm</span></div><input type="range" id="w" min="60" max="200" value="100" step="5" oninput="$('wV').textContent=this.value+'mm'"></div>
<div class="slider"><div class="slider-head"><label>Total Height</label><span class="val" id="hV">2.4mm</span></div><input type="range" id="h" min="1.6" max="4" value="2.4" step="0.2" oninput="$('hV').textContent=this.value+'mm';updateInstructions()"></div>
<div class="slider"><div class="slider-head"><label>Base Height</label><span class="val" id="bV">0.6mm</span></div><input type="range" id="b" min="0.4" max="1.2" value="0.6" step="0.1" oninput="$('bV').textContent=this.value+'mm';updateInstructions()"></div>
<div class="slider"><div class="slider-head"><label>Layer Height</label><span class="val" id="lV">0.12mm</span></div><input type="range" id="l" min="0.08" max="0.2" value="0.12" step="0.02" oninput="$('lV').textContent=this.value+'mm';updateInstructions()"></div>
</div>

<div class="sec">
<div class="sec-title">Image Processing</div>
<div class="slider"><div class="slider-head"><label>Contrast</label><span class="val" id="cV">1.5</span></div><input type="range" id="c" min="1" max="3" value="1.5" step="0.1" oninput="$('cV').textContent=this.value"></div>
<div class="slider"><div class="slider-head"><label>Denoise</label><span class="val" id="dV">2</span></div><input type="range" id="d" min="0" max="5" value="2" oninput="$('dV').textContent=this.value"></div>
</div>

<button class="btn btn-secondary" id="btnProcess" onclick="processImage()" disabled>Process Image</button>
<div class="btn-row">
<button class="btn btn-primary" id="btnSTL" onclick="exportSTL()" disabled>⬇ STL</button>
<button class="btn btn-secondary" id="btnPNG" onclick="exportPNG()" disabled>⬇ PNG</button>
</div>
<div class="status" id="status"></div>
</div>

<div class="preview-panel" id="panelPreview">
<div class="preview-header"><h2>Live Preview</h2><span class="badge" id="badge" style="display:none">Ready</span></div>
<div class="preview-area"><canvas id="canvas" style="display:none"></canvas><div id="placeholder">Upload an image to see live preview</div></div>
</div>

<div class="panel" id="panelInfo">
<div class="sec-title">Print Instructions</div>
<div class="instr" id="instr">1. Upload image
2. Choose number of colors
3. Pick your filament colors
4. Adjust thresholds with live preview
5. Export STL and print!</div>
<div class="sec">
<div class="sec-title">Model Info</div>
<div class="stats">
<div class="stat"><label>Width</label><div class="v" id="stW">-</div></div>
<div class="stat"><label>Height</label><div class="v" id="stH">-</div></div>
<div class="stat"><label>Layers</label><div class="v" id="stL">-</div></div>
</div>
</div>
<div class="footer">Made by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div>
</div>
</div>

<div class="mobile-tabs">
<div class="tabs">
<div class="tab active" onclick="showPanel('Settings')">Settings</div>
<div class="tab" onclick="showPanel('Preview')">Preview</div>
<div class="tab" onclick="showPanel('Info')">Info</div>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
let imgData=null,grayData=null,numColors=2;

// Default colors and thresholds for each layer count
const defaultColors = {
    2: [{color:'#FFFFFF',name:'Base (white)'},{color:'#222222',name:'Top (black)',thresh:128}],
    3: [{color:'#FFFFFF',name:'Base (white)'},{color:'#888888',name:'Mid (gray)',thresh:170},{color:'#222222',name:'Top (black)',thresh:85}],
    4: [{color:'#FFFFFF',name:'Base (white)'},{color:'#BBBBBB',name:'Light gray',thresh:190},{color:'#666666',name:'Dark gray',thresh:120},{color:'#222222',name:'Top (black)',thresh:60}]
};

let layers = JSON.parse(JSON.stringify(defaultColors[2]));

function renderLayerControls(){
    const container=$('layerControls');
    container.innerHTML='';
    layers.forEach((layer,i)=>{
        const isBase = i===0;
        const div=document.createElement('div');
        div.className='layer-row';
        div.innerHTML=`
            <input type="color" class="color-pick" value="${layer.color}" onchange="updateLayerColor(${i},this.value)">
            <div class="layer-info">
                <div class="layer-name">Layer ${i+1}${isBase?' (Base)':''}</div>
                <div class="layer-desc">${isBase?'Bottom layer - thinnest areas':layer.name}</div>
                ${!isBase?`<input type="range" min="5" max="250" value="${layer.thresh||128}" oninput="updateThreshold(${i},this.value)">`:''}
            </div>
        `;
        container.appendChild(div);
    });
}

function updateLayerColor(idx,color){
    layers[idx].color=color;
    updateLivePreview();
    updateInstructions();
}

function updateThreshold(idx,val){
    layers[idx].thresh=parseInt(val);
    updateLivePreview();
}

function setColorCount(n){
    numColors=n;
    document.querySelectorAll('.cc-btn').forEach((b,i)=>b.classList.toggle('active',i===n-2));
    layers=JSON.parse(JSON.stringify(defaultColors[n]));
    renderLayerControls();
    updateLivePreview();
    updateInstructions();
}

// File handling
const up=$('up');
up.ondragover=e=>{e.preventDefault();up.style.borderColor='#00AE42'};
up.ondragleave=()=>up.style.borderColor='#2a2a2a';
up.ondrop=e=>{e.preventDefault();up.style.borderColor='#2a2a2a';if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])};
$('file').onchange=e=>{if(e.target.files.length)handleFile(e.target.files[0])};

function handleFile(f){
    if(!f.type.startsWith('image/')){msg('error','Please upload an image');return}
    const reader=new FileReader();
    reader.onload=e=>{
        imgData=e.target.result;
        up.innerHTML='<img src="'+imgData+'">';
        up.classList.add('has-img');
        $('btnProcess').disabled=false;
        loadImageForPreview(imgData);
        msg('success','Image loaded');
    };
    reader.readAsDataURL(f);
}

function loadImageForPreview(src){
    const img=new Image();
    img.onload=()=>{
        const canvas=$('canvas');
        const maxSize=500;
        let w=img.width,h=img.height;
        if(w>maxSize||h>maxSize){const s=maxSize/Math.max(w,h);w=Math.round(w*s);h=Math.round(h*s)}
        canvas.width=w;canvas.height=h;
        const ctx=canvas.getContext('2d');
        ctx.drawImage(img,0,0,w,h);
        const imageData=ctx.getImageData(0,0,w,h);
        grayData=new Uint8Array(w*h);
        for(let i=0;i<w*h;i++){
            const r=imageData.data[i*4],g=imageData.data[i*4+1],b=imageData.data[i*4+2];
            grayData[i]=Math.round(0.299*r+0.587*g+0.114*b);
        }
        canvas.style.display='block';
        $('placeholder').style.display='none';
        updateLivePreview();
    };
    img.src=src;
}

function hexToRgb(hex){
    const r=parseInt(hex.slice(1,3),16);
    const g=parseInt(hex.slice(3,5),16);
    const b=parseInt(hex.slice(5,7),16);
    return [r,g,b];
}

function updateLivePreview(){
    if(!grayData)return;
    const canvas=$('canvas');
    const ctx=canvas.getContext('2d');
    const w=canvas.width,h=canvas.height;
    const imageData=ctx.createImageData(w,h);
    
    // Sort thresholds descending (brightest first)
    const sortedLayers=[...layers].map((l,i)=>({...l,idx:i}));
    // Base layer (0) has no threshold, others sorted by thresh descending
    const threshLayers=sortedLayers.slice(1).sort((a,b)=>(b.thresh||0)-(a.thresh||0));
    
    for(let i=0;i<w*h;i++){
        const g=grayData[i];
        let rgb=hexToRgb(layers[0].color); // default to base
        
        for(const layer of threshLayers){
            if(g < (layer.thresh||128)){
                rgb=hexToRgb(layer.color);
            }
        }
        
        imageData.data[i*4]=rgb[0];
        imageData.data[i*4+1]=rgb[1];
        imageData.data[i*4+2]=rgb[2];
        imageData.data[i*4+3]=255;
    }
    ctx.putImageData(imageData,0,0);
}

function updateInstructions(){
    const total=parseFloat($('h').value);
    const base=parseFloat($('b').value);
    const layer=parseFloat($('l').value);
    const totalLayers=Math.round(total/layer);
    const relief=total-base;
    
    $('stL').textContent=totalLayers;
    
    let instr=`Layer: ${layer}mm | Total: ${total}mm | ${totalLayers} layers\n\n`;
    instr+=`1. Start with ${layers[0].color.toUpperCase()}\n`;
    
    const threshLayers=layers.slice(1).map((l,i)=>({...l,origIdx:i+1})).sort((a,b)=>(b.thresh||0)-(a.thresh||0));
    
    threshLayers.forEach((l,i)=>{
        const fraction = 1 - ((l.thresh||128)/255);
        const z = base + relief * fraction;
        const layerNum = Math.round(z/layer);
        instr+=`${i+2}. Change to ${l.color.toUpperCase()} at layer ${layerNum} (Z≈${z.toFixed(1)}mm)\n`;
    });
    
    $('instr').textContent=instr;
}

async function processImage(){
    if(!imgData)return;
    $('btnProcess').disabled=true;
    $('btnProcess').textContent='Processing...';
    try{
        const r=await fetch('/api/print/process',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                image:imgData,
                num_colors:numColors,
                contrast:parseFloat($('c').value),
                denoise:parseFloat($('d').value),
                width:parseFloat($('w').value),
                total_height:parseFloat($('h').value),
                base_height:parseFloat($('b').value),
                layer_height:parseFloat($('l').value),
                border:2
            })
        });
        const d=await r.json();
        if(d.error)throw new Error(d.error);
        $('stW').textContent=d.model_width.toFixed(0)+'mm';
        $('stH').textContent=d.model_height.toFixed(0)+'mm';
        $('badge').style.display='inline';
        $('btnSTL').disabled=false;
        $('btnPNG').disabled=false;
        msg('success','Ready to export!');
    }catch(e){
        msg('error',e.message);
    }finally{
        $('btnProcess').disabled=false;
        $('btnProcess').textContent='Process Image';
    }
}

async function exportSTL(){
    $('btnSTL').disabled=true;
    $('btnSTL').textContent='...';
    try{
        const r=await fetch('/api/print/stl',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                image:imgData,
                contrast:parseFloat($('c').value),
                denoise:parseFloat($('d').value),
                width:parseFloat($('w').value),
                total_height:parseFloat($('h').value),
                base_height:parseFloat($('b').value),
                border:2
            })
        });
        if(!r.ok)throw new Error('Export failed');
        const blob=await r.blob();
        const a=document.createElement('a');
        a.href=URL.createObjectURL(blob);
        a.download='hueforge_print.stl';
        a.click();
        msg('success','STL downloaded!');
    }catch(e){
        msg('error',e.message);
    }finally{
        $('btnSTL').disabled=false;
        $('btnSTL').textContent='⬇ STL';
    }
}

function exportPNG(){
    const canvas=$('canvas');
    if(!canvas||!canvas.width)return;
    const a=document.createElement('a');
    a.href=canvas.toDataURL('image/png');
    a.download='hueforge_preview.png';
    a.click();
    msg('success','PNG downloaded!');
}

function showPanel(name){
    document.querySelectorAll('.panel,.preview-panel').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.mobile-tabs .tab').forEach(t=>t.classList.remove('active'));
    $('panel'+name).classList.add('active');
    event.target.classList.add('active');
}

function msg(type,text){
    const el=$('status');
    el.className='status '+type;
    el.textContent=text;
    setTimeout(()=>el.className='status',3000);
}

// Initialize
renderLayerControls();
updateInstructions();
</script>
</body></html>"""

MAP_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>Map to STL | SebGE Tools</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--g:#00AE42;--gd:#009639;--bg:#0a0a0a;--c:#151515;--c2:#1e1e1e;--br:#2a2a2a;--t:#fff;--t2:#888}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--t);height:100vh;overflow:hidden}
.app{display:grid;grid-template-columns:280px 1fr 260px;height:100vh}
.panel{background:var(--c);padding:12px;display:flex;flex-direction:column;gap:8px;overflow-y:auto}
.panel::-webkit-scrollbar{width:4px}
.panel::-webkit-scrollbar-thumb{background:var(--br);border-radius:2px}
.back{color:var(--t2);text-decoration:none;font-size:10px}
.back:hover{color:var(--g)}
.logo{display:flex;align-items:center;gap:8px;padding-bottom:8px;border-bottom:1px solid var(--br)}
.logo-icon{width:32px;height:32px;background:linear-gradient(135deg,var(--g),var(--gd));border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px}
.logo h1{font-size:13px;font-weight:700}
.logo small{display:block;font-size:7px;color:var(--g);font-weight:500;margin-top:2px}
.search{position:relative}
.search input{width:100%;padding:10px 10px 10px 32px;background:var(--c2);border:1px solid var(--br);border-radius:6px;color:var(--t);font-size:14px}
.search input:focus{outline:none;border-color:var(--g)}
.search svg{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--t2);width:14px;height:14px}
.search-btn{position:absolute;right:4px;top:50%;transform:translateY(-50%);background:var(--g);border:none;color:#fff;padding:6px 12px;border-radius:4px;font-size:10px;cursor:pointer}
.coords{display:flex;gap:6px;margin-top:4px}
.coord{flex:1;background:var(--c2);border-radius:4px;padding:6px 8px;font-size:10px}
.coord label{color:var(--t2);font-size:8px;text-transform:uppercase}
.coord span{color:var(--g);font-family:monospace;font-weight:600}
.sec{background:var(--c2);border-radius:8px;padding:10px}
.sec-title{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--g);margin-bottom:8px}
.modes{display:flex;gap:4px}
.mode{flex:1;padding:10px 4px;background:var(--c);border:2px solid var(--br);border-radius:6px;cursor:pointer;text-align:center;transition:all .15s}
.mode:hover,.mode.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.mode .icon{font-size:20px}
.mode .name{font-size:9px;font-weight:600;margin-top:2px}
.mode .desc{font-size:7px;color:var(--t2)}
.presets{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px}
.preset{padding:6px 10px;background:var(--c);border:1px solid var(--br);border-radius:4px;cursor:pointer;font-size:10px;transition:all .15s}
.preset:hover,.preset.active{border-color:var(--g);background:var(--g);color:#fff}
.street-types{display:grid;grid-template-columns:repeat(2,1fr);gap:4px}
.stype{display:flex;align-items:center;gap:4px;padding:6px 8px;background:var(--c);border:1px solid var(--br);border-radius:4px;cursor:pointer;font-size:9px;transition:all .15s}
.stype:hover,.stype.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.stype .icon{font-size:12px}
.stype .check{width:14px;height:14px;border:1px solid var(--br);border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:9px;margin-left:auto}
.stype.active .check{background:var(--g);border-color:var(--g);color:#fff}
.markers{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}
.marker{padding:8px 4px;background:var(--c);border:2px solid var(--br);border-radius:6px;cursor:pointer;text-align:center;font-size:16px;transition:all .15s}
.marker:hover,.marker.active{border-color:var(--g);background:rgba(0,174,66,.1)}
.marker span{display:block;font-size:8px;margin-top:2px}
.slider{margin-bottom:6px}
.slider-head{display:flex;justify-content:space-between;margin-bottom:3px}
.slider label{font-size:10px;color:var(--t2)}
.slider .val{font-size:10px;color:var(--g);font-family:monospace;font-weight:600}
.slider input{width:100%;height:6px;background:var(--c);border-radius:3px;-webkit-appearance:none}
.slider input::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;background:var(--g);border-radius:50%;cursor:pointer}
.btn{padding:12px;border:none;border-radius:6px;font-size:12px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;transition:all .15s;width:100%}
.btn-primary{background:linear-gradient(135deg,var(--g),var(--gd));color:#fff}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,174,66,.3)}
.btn-secondary{background:var(--c2);color:var(--t);border:1px solid var(--br)}
.btn-secondary:hover{border-color:var(--g);color:var(--g)}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.status{padding:8px;border-radius:6px;font-size:10px;display:none;text-align:center}
.status.error{display:block;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#ef4444}
.status.success{display:block;background:rgba(0,174,66,.1);border:1px solid rgba(0,174,66,.3);color:var(--g)}
.map-container{position:relative;background:#111}
#map{width:100%;height:100%}
.map-hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.8);padding:6px 14px;border-radius:20px;font-size:10px;color:var(--t2)}
#preview3d{flex:1;background:var(--c2);border-radius:8px;min-height:150px;display:flex;align-items:center;justify-content:center;color:var(--t2);font-size:11px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px}
.stat{background:var(--c2);border-radius:5px;padding:8px;text-align:center}
.stat label{font-size:8px;color:var(--t2);text-transform:uppercase}
.stat .val{font-size:12px;color:var(--g);font-family:monospace;font-weight:600;margin-top:2px}
.info{font-size:9px;color:var(--t2);line-height:1.4;background:var(--c2);border-radius:6px;padding:10px}
.info strong{color:var(--g)}
.footer{margin-top:auto;text-align:center;font-size:9px;color:var(--t2);padding-top:8px;border-top:1px solid var(--br)}
.footer a{color:var(--g);text-decoration:none}
.spinner{width:14px;height:14px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.marker-opts{margin-top:8px;padding-top:8px;border-top:1px solid var(--br)}
.street-sec{display:block}
.mobile-tabs{display:none;position:fixed;bottom:0;left:0;right:0;background:var(--c);border-top:1px solid var(--br);padding:8px;z-index:1000}
.mobile-tabs .tabs{display:flex;gap:4px}
.mobile-tabs .tab{flex:1;padding:12px 8px;text-align:center;background:var(--c2);border-radius:8px;font-size:11px;cursor:pointer;font-weight:500}
.mobile-tabs .tab.active{background:var(--g);color:#fff}
.mobile-tabs .tab .icon{font-size:16px;display:block;margin-bottom:2px}
@media(max-width:900px){
    body{height:auto;min-height:100vh;overflow:auto}
    .app{display:block;height:auto;padding-bottom:80px}
    .panel{display:none;min-height:calc(100vh - 80px)}
    .panel.active{display:flex}
    .map-container{display:none;height:50vh}
    .map-container.active{display:block}
    .mobile-tabs{display:block}
    .street-types{grid-template-columns:1fr}
}
</style></head><body>
<div class="app">
<div class="panel active" id="panelSettings">
<a href="/" class="back">← Back to Tools</a>
<div class="logo">
<div class="logo-icon">🗺️</div>
<div><h1>Map → STL</h1><small>3D PRINT YOUR WORLD</small></div>
</div>
<div class="search">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
<input type="text" id="loc" placeholder="Search location..." value="Dusseldorf" onkeypress="if(event.key==='Enter')search()">
<button class="search-btn" onclick="search()">Go</button>
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
<div class="status" id="status"></div>
</div>

<div class="map-container active" id="panelMap"><div id="map"></div><div class="map-hint">Tap map to set center</div></div>

<div class="panel" id="panelPreview">
<div class="sec-title">3D Preview</div>
<div id="preview3d">Tap Preview to load</div>
<div class="stats">
<div class="stat"><label>Mode</label><div class="val" id="statMode">Streets</div></div>
<div class="stat"><label>Data</label><div class="val" id="statData">-</div></div>
</div>
<div class="btns">
<button class="btn btn-secondary" id="btnPreview" onclick="preview()">Preview</button>
<button class="btn btn-primary" id="btnExport" onclick="exportSTL()">⬇ Export STL</button>
</div>
<div class="info"><strong>Multi-Color Tip:</strong> Use Gap to separate marker from streets for different colors in Bambu Studio!</div>
<div class="footer">Made by <a href="https://makerworld.com/de/@SebGE" target="_blank">@SebGE</a></div>
</div>
</div>

<div class="mobile-tabs">
<div class="tabs">
<div class="tab active" onclick="showPanel('Settings')"><span class="icon">⚙️</span>Settings</div>
<div class="tab" onclick="showPanel('Map')"><span class="icon">🗺️</span>Map</div>
<div class="tab" onclick="showPanel('Preview')"><span class="icon">👁️</span>Preview</div>
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
    L.control.zoom({position:'topright'}).addTo(map);
    updateMap();
    map.on('click',e=>{lat=e.latlng.lat;lon=e.latlng.lng;updateMap();updateCoords()});
    document.querySelectorAll('.stype').forEach(el=>{el.onclick=()=>{el.classList.toggle('active');document.querySelectorAll('.preset').forEach(p=>p.classList.remove('active'))}});
    init3D();
    setTimeout(()=>map.invalidateSize(),100);
}
function init3D(){
    const c=$('preview3d');scene=new THREE.Scene();scene.background=new THREE.Color(0x1e1e1e);
    camera=new THREE.PerspectiveCamera(45,c.clientWidth/Math.max(c.clientHeight,150),0.1,1000);camera.position.set(80,80,80);camera.lookAt(0,0,0);
    renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(c.clientWidth,Math.max(c.clientHeight,150));
    scene.add(new THREE.AmbientLight(0xffffff,0.6));const dir=new THREE.DirectionalLight(0xffffff,0.8);dir.position.set(50,100,50);scene.add(dir);
}
function updateMap(){const r=+$('rad').value;if(mapMarker)map.removeLayer(mapMarker);if(circle)map.removeLayer(circle);mapMarker=L.marker([lat,lon],{icon:L.divIcon({html:'<div style="width:14px;height:14px;background:#00AE42;border:3px solid #fff;border-radius:50%;box-shadow:0 2px 6px rgba(0,0,0,.3)"></div>',iconSize:[14,14],iconAnchor:[7,7]})}).addTo(map);circle=L.circle([lat,lon],{radius:r,color:'#00AE42',fillOpacity:.08,weight:2}).addTo(map)}
function updRad(){const r=$('rad').value;$('radV').textContent=r>=1000?(r/1000)+'km':r+'m';if(circle)circle.setRadius(+r)}
function updateCoords(){$('latV').textContent=lat.toFixed(5);$('lonV').textContent=lon.toFixed(5)}
function setMode(m){mode=m;document.querySelectorAll('.mode').forEach(el=>el.classList.toggle('active',el.dataset.mode===m));$('statMode').textContent=m.charAt(0).toUpperCase()+m.slice(1);document.querySelectorAll('.street-sec').forEach(el=>el.style.display=m==='streets'?'block':'none')}
function preset(name){const types=presets[name]||[];document.querySelectorAll('.stype').forEach(el=>el.classList.toggle('active',types.includes(el.dataset.t)));document.querySelectorAll('.preset').forEach(p=>p.classList.toggle('active',p.textContent.toLowerCase().includes(name)))}
function getSelectedTypes(){return Array.from(document.querySelectorAll('.stype.active')).map(el=>el.dataset.t)}
function setMarker(m){markerType=m;document.querySelectorAll('.marker').forEach(el=>el.classList.toggle('active',el.dataset.m===m));$('markerOpts').style.display=m==='none'?'none':'block'}
async function search(){const q=$('loc').value.trim();if(!q)return;try{const r=await fetch('/api/geocode?q='+encodeURIComponent(q));const d=await r.json();if(d.error)throw new Error(d.error);lat=d.lat;lon=d.lon;map.setView([lat,lon],14);updateMap();updateCoords();msg('success','Found: '+d.name.substring(0,25))}catch(e){msg('error',e.message)}}
async function preview(){const btn=$('btnPreview');btn.disabled=true;btn.innerHTML='<div class="spinner"></div>';try{const r=await fetch('/api/map/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon,radius:+$('rad').value,size:+$('size').value,height:+$('ht').value,lineWidth:+$('lw').value,mode,streetTypes:getSelectedTypes(),marker:markerType,markerSize:+$('ms').value,markerGap:+$('mg').value})});const d=await r.json();if(d.error)throw new Error(d.error);load3DPreview(d.vertices,d.faces);$('statData').textContent=d.count+(mode==='city'?' bldgs':' segs');msg('success','Preview loaded!')}catch(e){msg('error',e.message)}finally{btn.disabled=false;btn.innerHTML='Preview'}}
function load3DPreview(verts,faces){
    const container=$('preview3d');
    const w=container.clientWidth||300;
    const h=container.clientHeight||250;
    // Reinitialize renderer with correct size
    if(!renderer||renderer.domElement.width!==w){
        scene=new THREE.Scene();scene.background=new THREE.Color(0x1e1e1e);
        camera=new THREE.PerspectiveCamera(45,w/h,0.1,1000);camera.position.set(80,80,80);camera.lookAt(0,0,0);
        renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setSize(w,h);renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
        scene.add(new THREE.AmbientLight(0xffffff,0.6));const dir=new THREE.DirectionalLight(0xffffff,0.8);dir.position.set(50,100,50);scene.add(dir);
    }
    if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose()}
    const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));geom.setIndex(faces);geom.computeVertexNormals();
    const mat=new THREE.MeshPhongMaterial({color:0x00AE42,flatShading:true});mesh=new THREE.Mesh(geom,mat);
    geom.computeBoundingBox();const box=geom.boundingBox;const center=new THREE.Vector3();box.getCenter(center);mesh.position.sub(center);
    const maxDim=Math.max(box.max.x-box.min.x,box.max.y-box.min.y,box.max.z-box.min.z);const scale=60/maxDim;mesh.scale.set(scale,scale,scale);scene.add(mesh);
    container.innerHTML='';container.appendChild(renderer.domElement);
    let angle=0;function animate(){requestAnimationFrame(animate);angle+=0.005;camera.position.x=Math.sin(angle)*100;camera.position.z=Math.cos(angle)*100;camera.lookAt(0,0,0);renderer.render(scene,camera)}animate()
}
async function exportSTL(){const btn=$('btnExport');btn.disabled=true;btn.innerHTML='<div class="spinner"></div>';try{const r=await fetch('/api/map/stl',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon,radius:+$('rad').value,size:+$('size').value,height:+$('ht').value,lineWidth:+$('lw').value,mode,streetTypes:getSelectedTypes(),marker:markerType,markerSize:+$('ms').value,markerGap:+$('mg').value})});if(!r.ok){const err=await r.json();throw new Error(err.error||'Export failed')}const blob=await r.blob();const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='map_'+lat.toFixed(4)+'_'+lon.toFixed(4)+'.stl';a.click();msg('success','STL downloaded!')}catch(e){msg('error',e.message)}finally{btn.disabled=false;btn.innerHTML='⬇ Export STL'}}
function showPanel(name){
    document.querySelectorAll('.panel,.map-container').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.mobile-tabs .tab').forEach(t=>t.classList.remove('active'));
    if(name==='Map'){$('panelMap').classList.add('active');setTimeout(()=>map.invalidateSize(),100)}
    else{$('panel'+name).classList.add('active')}
    event.target.closest('.tab').classList.add('active');
    // Resize renderer when showing preview panel
    if(name==='Preview'&&renderer){
        setTimeout(()=>{
            const c=$('preview3d');
            if(c.clientWidth>0&&c.clientHeight>0){
                camera.aspect=c.clientWidth/c.clientHeight;camera.updateProjectionMatrix();
                renderer.setSize(c.clientWidth,c.clientHeight);
            }
        },150);
    }
}
function msg(type,text){const el=$('status');el.className='status '+type;el.textContent=text;setTimeout(()=>el.className='status',4000)}
document.addEventListener('DOMContentLoaded',init);
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
        marker=d.get('marker','none');ms=d.get('markerSize',12);mg=d.get('markerGap',2)
        street_types=d.get('streetTypes',None)
        if mode=='streets':
            gdf=fetch_streets(lat,lon,radius,street_types)
            mesh=create_streets_mesh(gdf,lat,lon,radius,size,height,lw,marker,ms,mg)
            count=len(gdf)
        elif mode=='city':mesh=create_city_mesh(lat,lon,radius,size,height,lw,1.0,1.5);count=len(fetch_buildings(lat,lon,radius))
        else:mesh=create_terrain_mesh(lat,lon,radius,size,height*5,1.5);count=6400
        return jsonify({'vertices':mesh.vertices.flatten().tolist(),'faces':mesh.faces.flatten().tolist(),'count':count})
    except Exception as e:
        import traceback;traceback.print_exc()
        return jsonify({'error':str(e)}),500

@app.route('/api/map/stl',methods=['POST'])
def api_map_stl():
    try:
        d=request.json;lat,lon,radius,size,height=d['lat'],d['lon'],d['radius'],d['size'],d['height']
        mode=d['mode'];lw=d.get('lineWidth',1.2)
        marker=d.get('marker','none');ms=d.get('markerSize',12);mg=d.get('markerGap',2)
        street_types=d.get('streetTypes',None)
        if mode=='streets':
            gdf=fetch_streets(lat,lon,radius,street_types)
            mesh=create_streets_mesh(gdf,lat,lon,radius,size,height,lw,marker,ms,mg)
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
        si=d.get('simplify',2);sm=d.get('smooth',0);inv=d.get('invert',False)
        sl=d.get('singleLine',False)
        if mode=='filled':svg,p,w,h=image_to_filled_svg(img_data,th,bl,si,sm,inv)
        else:svg,p,w,h=image_to_svg(img_data,th,bl,si,sm,inv,mode,sl)
        return jsonify({'svg':svg,'paths':p,'width':w,'height':h})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/print/analyze',methods=['POST'])
def api_print_analyze():
    try:
        d=request.json;img_b64=d['image']
        if ',' in img_b64:img_b64=img_b64.split(',')[1]
        return jsonify(analyze_print_image(base64.b64decode(img_b64)))
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
        buf1=io.BytesIO();Image.fromarray(gray).save(buf1,format='PNG')
        buf2=io.BytesIO();Image.fromarray(preview).save(buf2,format='PNG')
        return jsonify({'processed_image':base64.b64encode(buf1.getvalue()).decode(),'preview_image':base64.b64encode(buf2.getvalue()).decode(),'instructions':instr,'model_width':mw,'model_height':mh})
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
        return send_file(buf,mimetype='application/octet-stream',as_attachment=True,download_name='print.stl')
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/health')
def health():
    return jsonify({'status':'ok','version':'1.2','tools':['map','image','print']})

if __name__=='__main__':
    port=int(os.environ.get('PORT',8080))
    print(f"\n  SebGE Tools v1.2\n  http://localhost:{port}\n")
    app.run(host='0.0.0.0',port=port,debug=os.environ.get('DEBUG','false').lower()=='true')
