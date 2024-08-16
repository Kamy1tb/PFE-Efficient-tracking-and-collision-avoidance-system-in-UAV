import math
class Distance:
    def __init__(self):
        print("Distance object created")
        
    def haversine(self,lat1, lon1, lat2, lon2):
        R = 6371e3  # Rayon de la Terre en mètres

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c  # distance en mètres
        return distance

    def distance_between_drones(self,lat1, lon1, alt1, lat2, lon2, alt2):
        distance_horizontal = self.haversine(lat1, lon1, lat2, lon2)
        diff_altitude = abs(alt1 - alt2)
        distance_3d = math.sqrt(distance_horizontal**2 + diff_altitude**2)
        return distance_3d


