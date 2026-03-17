import lap
import numpy as np
from models.measurementmodels import RangeBearing
from models.measurementmodels_offset import RangeBearing as RangeBearingOffset

# IC = np.random.rand(4, 5)
# print(IC)
# print(lap.lapjv(IC, extend_cost=True))
offset = np.array([2.0, 0.0])
sigma_range = 0.2
sigma_bearing = np.deg2rad(1)

sensor = RangeBearing(sigma_range, sigma_bearing)
sensor_off = RangeBearingOffset(sigma_range, sigma_bearing, sensor_offset=offset)

test_state = np.array([0, 0, 0])
test_meas = np.array([2, 1])

print("h_ comparison:")
print(f"  sensor:     {sensor.h_(test_state, test_meas)}")
print(f"  sensor_off: {sensor_off.h_(test_state, test_meas)}")

print("\nH_x comparison:")
print(f"  sensor:\n {sensor.H_x(test_state, test_meas)}")
print(f"  sensor_off:\n {sensor_off.H_x(test_state, test_meas)}")

print("\nH_m comparison:")
print(f"  sensor:\n {sensor.H_m(test_state, test_meas)}")
print(f"  sensor_off:\n {sensor_off.H_m(test_state, test_meas)}")


