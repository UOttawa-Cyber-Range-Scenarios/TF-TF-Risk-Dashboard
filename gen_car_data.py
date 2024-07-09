import random
import json

# Predefined major cities in Canada with approximate coordinates
cities = {
    "Toronto": (43.65107, -79.347015),
    "Vancouver": (49.282729, -123.120738),
    "Montreal": (45.501689, -73.567256),
    "Calgary": (51.044733, -114.071883),
    "Ottawa": (45.421530, -75.697193),
    "Edmonton": (53.546124, -113.493823),
    "Quebec City": (46.813878, -71.207981),
    "Winnipeg": (49.895136, -97.138374)
}

ai_version = ['3.0.0','3.0.1','3.0.2','3.1.0','3.1.1','3.2.0','3.2.1']

ai_models = ['Griseo Autonomous Plus']*25 + ['Griseo Base Assist']*40 + ['Griseo Advanced Drive']*35 

# Function to generate random car data
def generate_car_data(num_cars=100):
    car_data = []
    for i in range(1,num_cars+1):
        city = random.choice(list(cities.keys()))
        lat, lon = cities[city]
        lat += random.uniform(-0.1, 0.1)
        lon += random.uniform(-0.1, 0.1)
        ai_ver = random.choice(ai_version)
        
        car = {
            "id": f"CAR{i:03d}",
            "latitude": lat,
            "longitude": lon,
            "car_model": random.choice(["Model G1", "Model G1 Sparks", "Model G2", "Model G2 Sport"]),
            "ai_model": ai_models[i-1],
            "ai_version": ai_ver,
            "miles_driven": random.randint(1000, 20000),
            "city": city,
            "system_upgrade": 'Up to Date' if ai_ver == '3.2.1' else random.choice(['Scheduled to Install','Pending Download'])
        }
        car_data.append(car)
    return car_data

# Generate the car data
car_data = generate_car_data()

# Write the car data to a file
with open('car_data.py', 'w') as file:
    file.write(f"car_data_dict = {json.dumps(car_data, indent=4)}")

print("Car data has been written to car_data.py")
