from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import requests, tempfile, math
import shapely.geometry as geom
import shapely.ops as ops
import trimesh
from xml.etree.ElementTree import Element, SubElement, ElementTree

app = Flask(__name__)
CORS(app)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def latlon_to_bbox(lat, lon, size_km):
    d = size_km / 111
    return {
        "min_lat": lat - d,
        "min_lon": lon - d,
        "max_lat": lat + d,
        "max_lon": lon + d
    }

def fetch_osm(query):
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=60)
    r.raise_for_status()
    return r.json()["elements"]

def extrude(polys, height):
    meshes = []
    for p in polys:
        try:
            meshes.append(trimesh.creation.extrude_polygon(p, height))
        except:
            pass
    return trimesh.util.concatenate(meshes) if meshes else None

# -------------------------------------------------
# OSM Data
# -------------------------------------------------

def get_roads(bbox, road_w, path_w):
    q = f"""
    [out:json];
    way["highway"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
    out body; >; out skel qt;
    """
    elements = fetch_osm(q)

    nodes = {}
    lines = []

    for e in elements:
        if e["type"] == "node":
            nodes[e["id"]] = (e["lon"], e["lat"])

    for e in elements:
        if e["type"] == "way":
            coords = [nodes[n] for n in e["nodes"] if n in nodes]
            if len(coords) > 1:
                width = path_w if e.get("tags", {}).get("highway") == "footway" else road_w
                lines.append(geom.LineString(coords).buffer(width))

    return lines

def get_buildings(bbox):
    q = f"""
    [out:json];
    way["building"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
    out body; >; out skel qt;
    """
    elements = fetch_osm(q)

    nodes = {}
    buildings = []

    for e in elements:
        if e["type"] == "node":
            nodes[e["id"]] = (e["lon"], e["lat"])

    for e in elements:
        if e["type"] == "way":
            coords = [nodes[n] for n in e["nodes"] if n in nodes]
            if len(coords) >= 3:
                poly = geom.Polygon(coords)
                levels = int(e.get("tags", {}).get("building:levels", 2))
                buildings.append((poly, levels * 3.0))

    return buildings

# -------------------------------------------------
# Marker
# -------------------------------------------------

def build_marker(marker, bbox):
    cx = (bbox["min_lon"] + bbox["max_lon"]) / 2
    cy = (bbox["min_lat"] + bbox["max_lat"]) / 2

    if marker == "heart":
        pts = [
            (cx, cy),
            (cx + 0.0003, cy + 0.0003),
            (cx, cy + 0.0006),
            (cx - 0.0003, cy + 0.0003)
        ]
        return trimesh.creation.extrude_polygon(geom.Polygon(pts), 1.5)

    if marker == "pin":
        return trimesh.creation.icosphere(radius=0.0004).apply_translation((cx, cy, 1))

    return None

# -------------------------------------------------
# Exporter
# -------------------------------------------------

def export_3mf(meshes):
    model = Element("model", {
        "unit": "millimeter",
        "xmlns": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    })
    resources = SubElement(model, "resources")
    build = SubElement(model, "build")

    idx = 1
    for mesh in meshes.values():
        obj = SubElement(resources, "object", {"id": str(idx), "type": "model"})
        m = SubElement(obj, "mesh")
        verts = SubElement(m, "vertices")
        tris = SubElement(m, "triangles")

        for v in mesh.vertices:
            SubElement(verts, "vertex", {"x": str(v[0]*1000), "y": str(v[1]*1000), "z": str(v[2]*1000)})

        for f in mesh.faces:
            SubElement(tris, "triangle", {"v1": str(f[0]), "v2": str(f[1]), "v3": str(f[2])})

        SubElement(build, "item", {"objectid": str(idx)})
        idx += 1

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".3mf")
    ElementTree(model).write(tmp.name)
    return tmp.name

# -------------------------------------------------
# API
# -------------------------------------------------

@app.route("/api/export", methods=["POST"])
def export():
    data = request.get_json()

    loc = data["location"]
    bbox = latlon_to_bbox(loc["lat"], loc["lon"], loc.get("size_km", 2))

    mode = data["mode"]
    export_type = data["export"]
    marker = data.get("marker", "none")

    road_cfg = data.get("roads", {})
    road_w = road_cfg.get("road_width", 0.00018)
    path_w = road_cfg.get("path_width", 0.0001)
    road_h = road_cfg.get("height", 0.8)

    meshes = {}

    # Base
    base = geom.box(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"])
    meshes["base"] = trimesh.creation.extrude_polygon(base, 2.5)

    # Roads
    road_polys = get_roads(bbox, road_w, path_w)
    meshes["roads"] = extrude(road_polys, road_h)

    # Buildings
    if mode in ["city", "terrain"]:
        b_meshes = []
        for poly, h in get_buildings(bbox):
            try:
                b_meshes.append(trimesh.creation.extrude_polygon(poly, h))
            except:
                pass
        if b_meshes:
            meshes["buildings"] = trimesh.util.concatenate(b_meshes)

    # Marker
    if marker != "none":
        m = build_marker(marker, bbox)
        if m:
            meshes["marker"] = m

    # Export
    if export_type == "stl":
        mesh = trimesh.util.concatenate(meshes.values())
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
        mesh.export(tmp.name)
        return send_file(tmp.name, as_attachment=True)

    if export_type == "3mf":
        path = export_3mf(meshes)
        return send_file(path, as_attachment=True)

    return jsonify({"error": "invalid export"}), 400


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(debug=True)
