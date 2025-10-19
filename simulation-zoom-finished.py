import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from shapely.geometry import Point
from shapely.ops import unary_union
from itertools import combinations
import math

# ---------- Configuration ----------
fig, ax = plt.subplots(figsize=(12, 10))

MAP_SIZE_M = 2000
RESOLUTION = 2
GRID_SIZE = MAP_SIZE_M // RESOLUTION
rng = np.random.default_rng()

# Visual sizes/colors
CENTER_STAGE1_EDGE = "yellow"
CENTER_STAGE1_SIZE = 80
CENTER_STAGE2_EDGE = "red"
CENTER_STAGE2_SIZE = 120
ORANGE_SIZE = 60
ORANGE_HIGHLIGHT_SIZE = 110
ORANGE_EDGE = "black"
RECT_FILL = "#0b66d1"      # car color
RECT_FACE_ALPHA = 0.9
RECT_EDGE_COLOR = "black"
RECT_EDGE_WIDTH = 2.2
RECT_W = 14
RECT_H = 8
ROBOT_ROUTE_COLOR = "#1f77b4"   # blue route (drawn instantly)
PERIMETER_COLOR = "red"
BEACON_CONNECTION_COLOR = "pink"

# Landcover thresholds in meters
THRESHOLDS_M = {"dark_green": 300, "medium_green": 250, "light_green": 250, "open": 150}

# ---------- State ----------
grids = []
last_click = None
selected_node = None        # (g_idx, d_idx, dot_idx)
selected_orange = None      # index into orange_points
triangulable_area_m2 = 0.0
zoom_enabled = False
orange_points = []          # list of (x_world, y_world)
rectangles = []             # list of {'x','y','w','h','route':[(x,y),...]}
# Note: pressing U draws route instantly; no animation

# ---------- Terrain / grid helpers ----------
def generate_elevation(grid_size, num_hills=10):
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    elevation = np.zeros_like(X)
    for _ in range(num_hills):
        cx, cy = rng.uniform(-1, 1, 2)
        height = rng.uniform(-0.1, 0.1)
        spread = rng.uniform(0.3, 0.6)
        hill = height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
        elevation += hill
    return (elevation - elevation.min()) / (elevation.max() - elevation.min())

def generate_forest(grid_size, elevation):
    forest = np.zeros((grid_size, grid_size))
    for _ in range(50):
        cx, cy = rng.integers(0, grid_size, 2)
        radius = rng.integers(150, 300)
        Y, X = np.ogrid[:grid_size, :grid_size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        patch = np.clip(1 - dist / radius, 0, 1)
        forest += patch * rng.uniform(0.7, 1.0)
    forest *= (1.4 - elevation)
    return (forest - forest.min()) / (forest.max() - forest.min())

def classify_landcover(forest):
    lc = np.full(forest.shape, "open", dtype=object)
    lc[forest > 0.1] = "light_green"
    lc[forest > 0.25] = "medium_green"
    lc[forest > 0.45] = "dark_green"
    return lc

def create_grid(offset_x=0, offset_y=0):
    elevation = generate_elevation(GRID_SIZE)
    forest = generate_forest(GRID_SIZE, elevation)
    landcover = classify_landcover(forest)
    rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    color_map = {
        "open": [1.0, 1.0, 0.6],
        "light_green": [0.7, 1.0, 0.7],
        "medium_green": [0.3, 0.8, 0.3],
        "dark_green": [0.0, 0.4, 0.0],
    }
    for key, color in color_map.items():
        rgb[landcover == key] = color
    return {"offset_x": offset_x, "offset_y": offset_y, "rgb": rgb, "landcover": landcover, "drops": []}

# ---------- Geometry / area ----------
def compute_outline_and_area(all_dots):
    if len(all_dots) < 3:
        return None, 0.0, []
    buffers = []
    for x, y, cat in all_dots:
        th_m = THRESHOLDS_M.get(cat, 150)
        buffers.append(Point(x * RESOLUTION, y * RESOLUTION).buffer(th_m))
    triangulable = []
    for combo in combinations(buffers, 3):
        inter = combo[0].intersection(combo[1]).intersection(combo[2])
        if not inter.is_empty:
            triangulable.append(inter)
    if not triangulable:
        return None, 0.0, []
    area_union = unary_union(triangulable)
    area_m2 = area_union.area
    holes_xy = []
    def ring_coords(ring):
        coords = list(ring.coords)
        xs = np.array([c[0] for c in coords]) / RESOLUTION
        ys = np.array([c[1] for c in coords]) / RESOLUTION
        return xs, ys
    if area_union.geom_type == "Polygon":
        for ring in area_union.interiors:
            holes_xy.append(ring_coords(ring))
    else:
        for geom in area_union.geoms:
            if geom.geom_type == "Polygon":
                for ring in geom.interiors:
                    holes_xy.append(ring_coords(ring))
    return area_union, area_m2, holes_xy

# ---------- Node sampling ----------
def sample_nodes_around(grid, cx, cy, n=30):
    radius_px = 500 // RESOLUTION
    dots = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(radius_px * 0.2, radius_px * 0.8)
        dx = int(r * np.cos(angle)); dy = int(r * np.sin(angle))
        x = np.clip(cx + dx, 0, GRID_SIZE - 1)
        y = np.clip(cy + dy, 0, GRID_SIZE - 1)
        cat = grid["landcover"][y, x]
        dots.append((x, y, cat))
    return dots

# ---------- Selection helpers ----------
def nearest_drop_under_click(x_click, y_click):
    tol = 4
    for g_idx, grid in enumerate(grids):
        ox, oy = grid["offset_x"], grid["offset_y"]
        for d_idx, drop in enumerate(grid["drops"]):
            if abs((drop["cx"] + ox) - x_click) <= tol and abs((drop["cy"] + oy) - y_click) <= tol:
                return g_idx, d_idx
    return None

def nearest_dot_at(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return None
    x_click, y_click = event.xdata, event.ydata
    hit_tol = 8.0
    best = None; best_d = float('inf')
    for g_idx, grid in enumerate(grids):
        ox, oy = grid["offset_x"], grid["offset_y"]
        for d_idx, drop in enumerate(grid["drops"]):
            dots = drop["dots"]
            if not dots:
                continue
            for i, (dx, dy, _) in enumerate(dots):
                x = dx + ox; y = dy + oy
                dist = np.hypot(x - x_click, y - y_click)
                if dist < best_d and dist <= hit_tol:
                    best_d = dist; best = (g_idx, d_idx, i)
    return best

def orange_connect_targets_including_cars(x_world, y_world):
    candidates = []
    # include red nodes
    for g_idx, grid in enumerate(grids):
        ox, oy = grid["offset_x"], grid["offset_y"]
        for d_idx, drop in enumerate(grid["drops"]):
            dots = drop["dots"]
            if not dots:
                continue
            for i, (dx, dy, cat) in enumerate(dots):
                node_x = dx + ox; node_y = dy + oy
                dist_m = np.hypot((node_x - x_world) * RESOLUTION, (node_y - y_world) * RESOLUTION)
                th_m = THRESHOLDS_M.get(cat, 150)
                if dist_m <= th_m:
                    candidates.append((dist_m, "node", g_idx, d_idx, i, node_x, node_y))
    # include rectangle centers as potential "targets" (car center)
    for ridx, rect in enumerate(rectangles):
        cx = rect['x'] + rect['w']/2; cy = rect['y'] + rect['h']/2
        dist_m = np.hypot((cx - x_world) * RESOLUTION, (cy - y_world) * RESOLUTION)
        if dist_m <= THRESHOLDS_M.get("open", 150):
            candidates.append((dist_m, "car", ridx, None, None, cx, cy))
    if not candidates:
        return []
    candidates.sort(key=lambda t: t[0])
    selected = candidates[:3]
    out = []
    for c in selected:
        _, typ, a, b, cidx, nx, ny = c
        out.append((typ, a, b, cidx, nx, ny))
    return out

def nearest_orange_at(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return None
    x_click, y_click = event.xdata, event.ydata
    hit_tol = 12.0
    best_idx = None; best_d = float('inf')
    for idx, (oxw, oyw) in enumerate(orange_points):
        d = np.hypot(oxw - x_click, oyw - y_click)
        if d < best_d and d <= hit_tol:
            best_d = d; best_idx = idx
    return best_idx

# ---------- Draw / redraw ----------
def redraw(ax):
    ax.clear()
    outline_polys = []
    holes_polys = []

    # draw terrain and nodes
    for g_idx, grid in enumerate(grids):
        ox, oy = grid["offset_x"], grid["offset_y"]
        ax.imshow(grid["rgb"], origin="upper", extent=[ox, ox + GRID_SIZE, oy + GRID_SIZE, oy], zorder=0)
        for d_idx, drop in enumerate(grid["drops"]):
            cx, cy, dots, stage = drop["cx"], drop["cy"], drop["dots"], drop["stage"]
            if stage == 1:
                ax.scatter(cx + ox, cy + oy, s=CENTER_STAGE1_SIZE, facecolors='none',
                           edgecolors=CENTER_STAGE1_EDGE, linewidths=1.8, zorder=6)
            else:
                ax.scatter(cx + ox, cy + oy, s=CENTER_STAGE2_SIZE, facecolors='none',
                           edgecolors=CENTER_STAGE2_EDGE, linewidths=1.5, zorder=4)
            if stage >= 2 and dots:
                xs = [d[0] + ox for d in dots]; ys = [d[1] + oy for d in dots]
                ax.scatter(xs, ys, c='red', s=40, zorder=3)
            if stage >= 3 and dots and len(dots) > 1:
                for i, (x1, y1, cat1) in enumerate(dots):
                    for j, (x2, y2, cat2) in enumerate(dots):
                        if j <= i: continue
                        dist_m = math.hypot((x2 - x1) * RESOLUTION, (y2 - y1) * RESOLUTION)
                        th_m = min(THRESHOLDS_M.get(cat1,150), THRESHOLDS_M.get(cat2,150))
                        if dist_m <= th_m:
                            ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color=BEACON_CONNECTION_COLOR, linewidth=0.6, zorder=2)

    # stage4 perimeter
    all_stage4_dots = []
    for g_idx, grid in enumerate(grids):
        ox, oy = grid["offset_x"], grid["offset_y"]
        for drop in grid["drops"]:
            if drop["stage"] >= 4 and drop["dots"]:
                all_stage4_dots.extend([(d[0] + ox, d[1] + oy, d[2]) for d in drop["dots"]])
    global triangulable_area_m2
    if all_stage4_dots:
        outline, triangulable_area_m2, holes_xy = compute_outline_and_area(all_stage4_dots)
        if outline:
            polys = [outline] if outline.geom_type == 'Polygon' else [g for g in outline.geoms if g.geom_type == 'Polygon']
            for poly in polys:
                exterior_coords = list(poly.exterior.coords)
                xs = np.array([c[0] for c in exterior_coords]) / RESOLUTION
                ys = np.array([c[1] for c in exterior_coords]) / RESOLUTION
                ax.plot(xs, ys, color=PERIMETER_COLOR, linewidth=1.8, zorder=5)
                outline_polys.append((xs, ys))
            for xs, ys in holes_xy:
                ax.plot(xs, ys, color='darkblue', linewidth=1.5, zorder=6)
                holes_polys.append((xs, ys))
    else:
        triangulable_area_m2 = 0.0

    # draw orange points and their connections (connections may include cars)
    for idx, (ox_world, oy_world) in enumerate(orange_points):
        if selected_orange == idx:
            ax.scatter(ox_world, oy_world, c='orange', s=ORANGE_HIGHLIGHT_SIZE, zorder=9, edgecolors=ORANGE_EDGE, linewidths=2.4)
        else:
            ax.scatter(ox_world, oy_world, c='orange', s=ORANGE_SIZE, zorder=8, edgecolors=ORANGE_EDGE, linewidths=0.8)
        targets = orange_connect_targets_including_cars(ox_world, oy_world)
        # show connections only if exactly 3 targets available
        if len(targets) == 3:
            for typ, a, b, cidx, nx, ny in targets:
                if typ == "node":
                    ax.plot([ox_world, nx], [oy_world, ny], color='orange', linewidth=1.4, zorder=7)
                elif typ == "car":
                    ax.plot([ox_world, nx], [oy_world, ny], color='orange', linewidth=1.4, linestyle='--', zorder=7)

    # draw rectangles (cars) and any instant routes
    for ridx, rect in enumerate(rectangles):
        r = Rectangle((rect['x'], rect['y']), rect['w'], rect['h'],
                      linewidth=RECT_EDGE_WIDTH, edgecolor=RECT_EDGE_COLOR, facecolor=RECT_FILL,
                      alpha=RECT_FACE_ALPHA, zorder=10)
        ax.add_patch(r)
        route = rect.get('route', [])
        if route:
            xs = [p[0] for p in route]; ys = [p[1] for p in route]
            ax.plot(xs, ys, color=ROBOT_ROUTE_COLOR, linewidth=2.2, zorder=11)

    # legend (includes perimeter red line meaning)
    lg = ax.get_legend()
    if lg:
        lg.remove()
    legend_handles = [
        Patch(facecolor="yellow", edgecolor="k", label="Yellow center (no trees)"),
        Patch(facecolor=[0.7,1.0,0.7], edgecolor="k", label="Light green (light forest)"),
        Patch(facecolor=[0.3,0.8,0.3], edgecolor="k", label="Medium green (forest)"),
        Patch(facecolor=[0.0,0.4,0.0], edgecolor="k", label="Dark green (dense forest)"),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Red = beacon node'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=8, markeredgecolor='k', label='Orange = anchor node'),
        Patch(facecolor=RECT_FILL, edgecolor='k', label='Car (rectangle)'),
        Line2D([0],[0], color=ROBOT_ROUTE_COLOR, lw=2, label='Blue line = car route (instant)'),
        Line2D([0],[0], color=BEACON_CONNECTION_COLOR, lw=1, label='Pink lines = beacon connections'),
        Line2D([0],[0], color=PERIMETER_COLOR, lw=1.8, label='Red line = perimeter of triangulation area'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 0.98), framealpha=0.95)

    # view limits
    if zoom_enabled:
        xs_all, ys_all = [], []
        for g_idx, grid in enumerate(grids):
            ox, oy = grid["offset_x"], grid["offset_y"]
            for drop in grid["drops"]:
                if drop["stage"] >= 2 and drop["dots"]:
                    xs_all.extend([d[0] + ox for d in drop["dots"]])
                    ys_all.extend([d[1] + oy for d in drop["dots"]])
        for xs, ys in outline_polys + holes_polys:
            xs_all.extend(xs); ys_all.extend(ys)
        for ox_world, oy_world in orange_points:
            xs_all.append(ox_world); ys_all.append(oy_world)
        for rect in rectangles:
            xs_all.extend([rect['x'], rect['x'] + rect['w']])
            ys_all.extend([rect['y'], rect['y'] + rect['h']])
        if xs_all and ys_all:
            margin = 20
            ax.set_xlim(min(xs_all)-margin, max(xs_all)+margin)
            ax.set_ylim(max(ys_all)+margin, min(ys_all)-margin)
        else:
            ax.set_xlim(0, GRID_SIZE); ax.set_ylim(GRID_SIZE, 0)
    else:
        min_x = min((g['offset_x'] for g in grids), default=0)
        max_x = max((g['offset_x'] + GRID_SIZE for g in grids), default=GRID_SIZE)
        min_y = min((g['offset_y'] for g in grids), default=0)
        max_y = max((g['offset_y'] + GRID_SIZE for g in grids), default=GRID_SIZE)
        ax.set_xlim(min_x, max_x); ax.set_ylim(max_y, min_y)

    ax.text(0.02, 1.02, f"Triangulable Area (stage4): {triangulable_area_m2/1e6:.3f} km²", transform=ax.transAxes, fontsize=12, color='darkred', ha='left')
    ax.set_xticks([]); ax.set_yticks([])
    plt.draw()

# ---------- Interaction ----------
def on_click(event):
    global last_click, selected_node, selected_orange
    if event.inaxes != ax:
        return
    last_click = (event.xdata, event.ydata)
    o_idx = nearest_orange_at(event)
    if o_idx is not None:
        selected_orange = o_idx
        selected_node = None
        redraw(ax)
        return
    sel = nearest_dot_at(event)
    if sel is not None:
        selected_node = sel
        selected_orange = None
        redraw(ax)
        return
    selected_node = None; selected_orange = None
    redraw(ax)

def reset_all(ev=None):
    global grids, orange_points, rectangles, selected_node, selected_orange, triangulable_area_m2
    grids = []
    orange_points = []
    rectangles = []
    selected_node = None
    selected_orange = None
    triangulable_area_m2 = 0.0
    grids.append(create_grid(0,0))
    redraw(ax)

def on_key(event):
    global last_click, zoom_enabled, selected_orange, selected_node
    if event.key is None:
        return
    key = event.key.lower()

    if key == 'z':
        zoom_enabled = not zoom_enabled
        redraw(ax); return

    if key == 'd':
        if last_click is None: return
        x_click, y_click = last_click
        for g_idx, grid in enumerate(grids):
            ox, oy = grid["offset_x"], grid["offset_y"]
            if not (ox <= x_click < ox + GRID_SIZE and oy <= y_click < oy + GRID_SIZE): continue
            cx = int(x_click - ox); cy = int(y_click - oy)
            found = nearest_drop_under_click(x_click, y_click)
            if found:
                fg, fd = found
                drop = grids[fg]["drops"][fd]
                if drop["stage"] < 4:
                    drop["stage"] += 1
                    if drop["stage"] == 2:
                        drop["dots"] = sample_nodes_around(grid, drop["cx"], drop["cy"], n=30)
                redraw(ax)
                break
            else:
                grids[g_idx]["drops"].append({"cx": cx, "cy": cy, "dots": [], "stage": 1})
                redraw(ax)
                break
        return

    if key == 'o':
        if last_click is None: return
        orange_points.append(last_click)
        selected_orange = len(orange_points)-1
        redraw(ax); return

    if key == 'p':
        if selected_orange is not None and 0 <= selected_orange < len(orange_points):
            orange_points.pop(selected_orange)
            selected_orange = None
            redraw(ax)
        return

    if key == 'r':
        if selected_node is not None:
            g_idx, d_idx, i = selected_node
            if 0 <= g_idx < len(grids) and 0 <= d_idx < len(grids[g_idx]["drops"]):
                dots = grids[g_idx]["drops"][d_idx]["dots"]
                if 0 <= i < len(dots):
                    dots.pop(i)
                    selected_node = None
                    redraw(ax)
        return

    if key == 'c':
        if last_click is None: return
        xw, yw = last_click
        w = RECT_W; h = RECT_H
        x0 = xw - w / 2; y0 = yw - h / 2
        rectangles.append({'x': x0, 'y': y0, 'w': w, 'h': h, 'route': []})
        redraw(ax); return

    if key == 'u':
        # draw instant straight blue line from nearest car to selected orange anchor
        if selected_orange is None or not orange_points or not rectangles:
            return
        target = orange_points[selected_orange]
        best_idx = None; best_dist = float('inf')
        for i, rect in enumerate(rectangles):
            cx = rect['x'] + rect['w']/2; cy = rect['y'] + rect['h']/2
            d = math.hypot(cx - target[0], cy - target[1])
            if d < best_dist:
                best_dist = d; best_idx = i
        if best_idx is None: return
        rect = rectangles[best_idx]
        start = (rect['x'] + rect['w']/2, rect['y'] + rect['h']/2)
        goal = target
        # set the route to just the two endpoints (straight line)
        rect['route'] = [start, goal]
        redraw(ax); return

    if key == 'x':
        reset_all(); return

# ---------- Utility helpers ----------
def drop_nodes_on_grid(grid, cx, cy, n=30):
    dots = sample_nodes_around(grid, cx, cy, n=n)
    grid["drops"].append({"cx": cx, "cy": cy, "dots": dots, "stage": 2})
    return dots

# ---------- Initial UI and bindings ----------
grids.append(create_grid(0,0))
redraw(ax)

button_ax = plt.axes([0.66,0.02,0.30,0.05])
button = Button(button_ax, "Generate New Map (Reset)")
button.on_clicked(reset_all)

add_ax = plt.axes([0.36,0.02,0.28,0.05])
add_button = Button(add_ax, "Add Adjacent Grid")
add_button.on_clicked(lambda ev: (grids.append(create_grid(len(grids)*GRID_SIZE,0)), redraw(ax)))

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()