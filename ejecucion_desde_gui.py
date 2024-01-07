from absorcion_10534_2 import Absorcion_10534_2
import pyaudio
import matplotlib.pyplot as plt


# 1 - Desde la GUI se inicializa una nueva medición y se definen algunas variables/constantes

folder_name = 'prueba 2'
speaker_mic_distance = 0.283 
mic_distance = 75e-3
sound_speed = 340
air_density = 1.18
sample_name = 'nombre de prueba'
sample_description = 'hola'

# 2 - Se instancia un objeto Absorcion_10534_2

medicion = Absorcion_10534_2(folder_name = folder_name,
                             speaker_mic_distance = speaker_mic_distance,
                             mic_distance = mic_distance,
                             sound_speed = sound_speed,
                             air_density = air_density,
                             sample_name = sample_name,
                             sample_description = sample_description)

carpeta_medicion = medicion.folder_path

# 3 - Desde la GUI, el usuario elige el hardware y también configura otras variables/constantes
lista_hardware = list(medicion.devices.keys())

input_device_gui = 'Analogue 1 + 2 (Focusrite Usb A'    # 'MacBook Pro Microphone'
output_device_gui = 'Altavoces (Focusrite Usb Audio)'   # 'MacBook Pro Speakers'

buffer_size = 2048
format = pyaudio.paInt16  # Formato de audio (16 bits PCM)
channels_output = 1  # Número de canales (1 para mono, 2 para estéreo)
channels_input = 2
sample_rate = 44100  # Tasa de muestreo en Hz
sweep_lower_frequency = 50
sweep_upper_frequency = 5000
sweep_duration = 4
record_duration = sweep_duration
# Alpha calculation constants
smooth_response = True
smooth_window_size = 100
octave_fraction = 3

medicion.measurement_setup(input_device_gui = input_device_gui,
                           output_device_gui = output_device_gui,
                           buffer_size = buffer_size,
                           format = format,
                           channels_output =channels_output,
                           channels_input = channels_input,
                           sample_rate = sample_rate,
                           sweep_lower_frequency = sweep_lower_frequency,
                           sweep_upper_frequency = sweep_upper_frequency,
                           sweep_duration = sweep_duration,
                           record_duration = record_duration,
                           smooth_response = smooth_response,
                           smooth_window_size = smooth_window_size,
                           octave_fraction = octave_fraction)


# 4 - Se genera el sweep

medicion.generate_sweep()

# 5 - Se mide relación señal a ruido

snr_id = 1 # Este id debe ir aumentando si la medición se repite, para que quede registro
medicion.measure_snr(snr_id)

medicion.snr_data["freq_noise_A"]
medicion.snr_data["noise_frac_A"]
medicion.snr_data["noise_frac_B"]
medicion.snr_data["signal_frac_A"]
medicion.snr_data["signal_frac_B"]
medicion.snr_data["snr_A"]
medicion.snr_data["snr_B"]


# 6 - Se mide la Calibración AB - BA
cal_id = 1
medicion.measure_calibration(cal_id, "AB")
medicion.measure_calibration(cal_id, "BA")

# Se calcula la calibración y se verifica si todo está ok visualmente, sino repetir el proceso
medicion.calibration_calculation(cal_id)

frequency = medicion.correction_data["frequency"]
H1_corr_1 = medicion.correction_data["H1_corr_1"]
H1_corr_2 = medicion.correction_data["H1_corr_2"]
Hc = medicion.correction_data["Hc"]
Hc_smooth = medicion.correction_data["Hc_smooth"]

# 7 - Se mide absorcion N Veces

abs_ids = [1, 2]
for id in abs_ids:
    medicion.measure_absorption(id)
    medicion.absorption_coefficient_calculation(id)

frecuencia = medicion.absorption_data[1]["frequency"]
alpha = medicion.absorption_data[1]["alpha_coeff"]
r = medicion.absorption_data[1]["r_coeff"]
z = medicion.absorption_data[1]["z_coeff"]
frec_center = medicion.absorption_data[1]["freq_center"]
alpha_center = medicion.absorption_data[1]["alpha_frac"]
tf_corr = medicion.absorption_data[1]["tf"]

# plt.plot(frecuencia, tf_corr)
plt.plot(frec_center, alpha_center)
plt.show()
print('hola')

