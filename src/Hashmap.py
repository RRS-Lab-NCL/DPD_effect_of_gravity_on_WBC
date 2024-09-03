import math

def from_points(cell_size, points):
    """
    Build a HashMap from a list of points.
    Source from: https://www.pygame.org/wiki/SpatialHashMap
    """
    hashmap = HashMap(cell_size)
    
    for point in points:         
        idx = int(point[-1])
        # dict_setdefault(hashmap.grid, hashmap.key(point),[]).append(point)
        dict_setdefault(hashmap.grid, hashmap.key(point),[]).append(idx)    
    return hashmap

def dict_setdefault(D, k, d):        
    
    r = D.get(k, d)    
    if k not in D:
        D[k] = d
    
    return r

class HashMap(object):
    
    """
    Hashmap is a a spatial index which divides the space into certain area 
    and then stores all the particles containing in it.
    """
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}    

    def key(self, point):
        cell_size = self.cell_size
        return (int((math.floor(point[0]/cell_size))*cell_size),
                int((math.floor(point[1]/cell_size))*cell_size))

    def insert(self, point):
        """
        Useful for inserting a/collection of point into the hashmap.
        """
        dict_setdefault( self.grid, self.key(point), []).append(point)
   
    def query(self, point):
        """
        Return all objects in the cell specified by point.
        """            
        return dict_setdefault( self.grid, self.key(point), [])
    
    def query_radius(self, point, radius):
        """
        Return all objects within a given radius of the point.
        """
                    
        cell_size = int(self.cell_size)
        
        x, y = point[0], point[1]
        min_x = int((x - radius) // cell_size) * cell_size
        max_x = int((x + radius) // cell_size) * cell_size
        min_y = int((y - radius) // cell_size) * cell_size
        max_y = int((y + radius) // cell_size) * cell_size

        results = []
        for x_pos in range(min_x, max_x + cell_size, cell_size):
            for y_pos in range(min_y, max_y + cell_size, cell_size): 
                
                cell_objects = self.grid.get((x_pos, y_pos), [])
                results.extend(cell_objects)                                    # if (x - x_pos)**2 + (y - y_pos)**2 <= radius**2:
                    
        return results