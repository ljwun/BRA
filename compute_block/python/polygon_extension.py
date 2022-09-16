import numpy as np
from shapely.geometry import MultiPoint

def PolygonExtension(xys, ext_pixel):
    ori_zone = [(p['x'], p['y']) for p in xys]
    centroid = MultiPoint(ori_zone).convex_hull.centroid
    cx, cy = centroid.x, centroid.y
    out_vector = np.asarray([(p['x']-cx, p['y']-cy)for p in xys])
    out_vector = out_vector / np.linalg.norm(out_vector, axis=1, keepdims=True) * ext_pixel
    ext_zone = np.asarray(ori_zone) + out_vector
    return [{'x':xy[0], 'y':xy[1]} for xy in ext_zone]
