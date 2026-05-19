from geopy.geocoders import Nominatim
import pandas as pd
import time


class MunicipalityGeocoder:

    def __init__(self):

        self.geolocator = Nominatim(
            user_agent="cycling_project"
        )

    def geocode_municipalities(self, municipalities):

        results = []

        for municipality in municipalities:

            print(f"Geocoding {municipality}")

            try:

                location = self.geolocator.geocode(
                    f"{municipality}, Belgium"
                )

                if location:

                    results.append({
                        "municipality": municipality,
                        "latitude": location.latitude,
                        "longitude": location.longitude
                    })

                else:

                    results.append({
                        "municipality": municipality,
                        "latitude": None,
                        "longitude": None
                    })

            except Exception as e:

                print(e)

                results.append({
                    "municipality": municipality,
                    "latitude": None,
                    "longitude": None
                })

            time.sleep(1)

        return pd.DataFrame(results)