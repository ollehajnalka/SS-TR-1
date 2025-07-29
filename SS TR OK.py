import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Naprendszer adatai ---
x_data = np.array([57.9, 108.2, 149.6, 228, 778.5, 1432, 2867, 4515, 5906.4])  # 10^6 km
y_data = np.array([47.4, 35, 29.8, 24.1, 13.1, 9.7, 6.8, 5.4, 4.7])  # km/s

# --- 2. HD10180 adatok ---
# távolság AU-ban -> millió km
#z_data = np.array([0.06412, 0.1286, 0.2699, 0.4929, 1.427, 3.381]) * 149.6  # 10^6 km  HD10180
#T_days = np.array([5.7597, 16.3570, 49.748, 122.744, 604.67, 2205.0])  # nap HD10180
z_data = np.array([11.11, 15.21, 21.44, 28.17, 37.1, 45.1, 63]) * 149.6 * 1e-3  # 10^6 km TRAPPIST-1
T_days = np.array([1.51, 2.42, 4.05, 6.1, 9.2, 12.35, 20])  # nap TRAPPIST-1

# keringési sebesség km/s
w_data = 2 * np.pi * z_data * 1e6 / (T_days * 86400) / 1000  # átváltás: km/s

# --- 3. Illesztendő függvény ---
def func(x, a):
    return a / np.sqrt(x)

# --- 4. Illesztés ---
popt1, pcov1 = curve_fit(func, x_data, y_data)
a_fit = popt1[0]
a_err = np.sqrt(pcov1[0][0])

popt2, pcov2 = curve_fit(func, z_data, w_data)
b_fit = popt2[0]
b_err = np.sqrt(pcov2[0][0])

# --- 5. Görbe megjelenítés ---
x_fit = np.linspace(x_data.min(), x_data.max(), 100)
y_fit = func(x_fit, a_fit)
z_fit = np.linspace(z_data.min(), z_data.max(), 100)
w_fit = func(z_fit, b_fit)

plt.figure(figsize=(10, 6))
plt.scatter(z_data, w_data, label='Bolygók', color='b')
plt.plot(z_fit, w_fit, label='Illesztett görbe', color='r')
plt.title('TRAPPIST-1 rotációs görbéje')
plt.xlabel('Csillagtól való távolság (x$10^{6}$ km)')
plt.ylabel('Keringési sebesség (km/s)')
plt.legend()
plt.grid()
plt.show()

# --- 6. Tömegszámítás ---
G = 6.67430e-11  # m^3 / kg / s^2
# km/s * sqrt(10^6 km) -> m^(3/2)/s
def convert_param(p, p_err):
    p_SI = p * 1e3 * np.sqrt(1e9)
    p_SI_err = p_err * 1e3 * np.sqrt(1e9)
    return p_SI, p_SI_err

a_SI, a_SI_err = convert_param(a_fit, a_err)
b_SI, b_SI_err = convert_param(b_fit, b_err)

M_sun = a_SI**2 / G
M_sun_err = M_sun * 2 * a_SI_err / a_SI

M_trappist = b_SI**2 / G
M_trappist_err = M_trappist * 2 * b_SI_err / b_SI

# --- 7. Eredmények kiírása ---
print("\n--- Eredmények ---")
print(f"Naprendszer illesztett paraméter: a = {a_fit:.3f} ± {a_err:.3f} (km/s * sqrt(10^6 km))")
print(f"Naprendszer központi tömege: ({M_sun:.2e} ± {M_sun_err:.2e}) kg")

print(f"\nHD10180 illesztett paraméter: b = {b_fit:.3f} ± {b_err:.3f} (km/s * sqrt(10^6 km))")
print(f"HD10180 központi tömege: ({M_trappist:.2e} ± {M_trappist_err:.2e}) kg")
