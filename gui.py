
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Toplevel, Label
from absorcion_10534_2 import Absorcion_10534_2
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\MAITE\Documents\UNTREF\Tesis\Procesamiento\GitHub\Tesis\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1215x700")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 700,
    width = 1215,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    1215.0,
    101.0,
    fill="#013088",
    outline="")

canvas.create_text(
    264.0,
    10.0,
    anchor="nw",
    text="ISO 10534-2: Coeficiente de absorción sonora",
    fill="#FFFFFF",
    font=("Encode Sans", 40 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    122.0,
    49.0,
    image=image_image_1
)

canvas.create_rectangle(
    20.0,
    234.0,
    1202.0,
    235.0,
    fill="#000000",
    outline="")

canvas.create_rectangle(
    20.0,
    324.0,
    1202.0,
    325.0,
    fill="#000000",
    outline="")

canvas.create_rectangle(
    20.0,
    448.0,
    1202.0,
    449.0,
    fill="#000000",
    outline="")

canvas.create_rectangle(
    19.0,
    585.0,
    1201.0,
    586.0,
    fill="#000000",
    outline="")

canvas.create_text(
    28.0,
    114.0,
    anchor="nw",
    text="Realiza la medición:",
    fill="#000000",
    font=("Encode Sans", 20 * -1)
)

canvas.create_text(
    27.52734375,
    166.32839965820312,
    anchor="nw",
    text="Muestra:",
    fill="#000000",
    font=("Encode Sans", 20 * -1)
)

canvas.create_text(
    529.0,
    166.0,
    anchor="nw",
    text="Temperatura:",
    fill="#000000",
    font=("Encode Sans", 20 * -1)
)

canvas.create_text(
    789.0,
    165.0,
    anchor="nw",
    text="°C",
    fill="#000000",
    font=("Encode Sans", 20 * -1)
)

# canvas.create_text(
#     895.0,
#     205.0,
#     anchor="nw",
#     text="Registro de medición creado con éxito",
#     fill="green",
#     font=("Encode Sans", 16 * -1)
# )

text_snr = canvas.create_text(
    50.0,
    245.0,
    anchor="nw",
    text="Medición de relación \nseñal a ruido ",
    fill="#000000",
    font=("Encode Sans", 20 * -1),
    state="hidden"
)

text_result = canvas.create_text(
    626.0,
    260.0,
    anchor="nw",
    text="Resultado:",
    fill="#000000",
    font=("Encode Sans", 20 * -1),
    state="hidden"
)

text_snr_results = canvas.create_text(
    766.0,
    260.0,
    anchor="nw",
    text=" ",
    fill="#000000",
    font=("Encode Sans", 16 * -1)
)

text_correction = canvas.create_text(
    50.0,
    345.0,
    anchor="nw",
    text="Corrección de \nmicrófonos",
    fill="#000000",
    font=("Encode Sans", 20 * -1),
    state="hidden"
)

text_absorption = canvas.create_text(
    50.0,
    462.0,
    anchor="nw",
    text="Obtención de\ncoeficiente de \nabsorción sonora",
    fill="#000000",
    font=("Encode Sans", 20 * -1),
    state="hidden"
)

text_repetition = canvas.create_text(
    293.0,
    460.0,
    anchor="nw",
    text="Repeticiones:",
    fill="#000000",
    font=("Encode Sans", 20 * -1),
    state="hidden"
)

text_comments = canvas.create_text(
    93.0,
    592.0,
    anchor="nw",
    text="Observaciones:",
    fill="#000000",
    font=("Encode Sans", 16 * -1),
    state="hidden"
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    422.5,
    648.5,
    image=entry_image_1,
    state = "hidden"
)
entry_comments = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_comments.place_forget()
# entry_comments.place(
#     x=85.0,
#     y=622.0,
#     width=675.0,
#     height=51.0
# )

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    473.66015625,
    481.0122547149658,
    image=entry_image_2,
    state = "hidden"
)
entry_repetitions = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_repetitions.place_forget()
# entry_repetitions.place(
#     x=431.89453125,
#     y=466.0,
#     width=83.53125,
#     height=28.02450942993164
# )

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    726.630859375,
    185.01225471496582,
    image=entry_image_3
)
entry_temperature = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_temperature.place(
    x=673.0,
    y=170.0,
    width=107.26171875,
    height=28.02450942993164
)

entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    292.833984375,
    184.5,
    image=entry_image_4,
    state = "hidden"
)
entry_sample = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)

entry_sample.place(
    x=115.8046875,
    y=170.0,
    width=354.05859375,
    height=27.0
)

entry_image_5 = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_5 = canvas.create_image(
    488.578125,
    134.01225471496582,
    image=entry_image_5,
    state = "hidden"
)
entry_people = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_people.place(
    x=219.0,
    y=119.0,
    width=539.15625,
    height=28.02450942993164
)

button_image_1 = PhotoImage(                                # INFO
    file=relative_to_assets("button_1.png"))
button_info = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: infoButton(),
    relief="flat"
)
button_info.place(
    x=1122.0,
    y=16.0,
    width=69.0,
    height=64.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_measure = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: measureButton(),
    relief="flat"
)
button_measure.place(
    x=898.0,
    y=119.0,
    width=266.72662353515625,
    height=89.16289520263672
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_snr = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: snrButton(medicion),
    relief="flat"
)
button_snr.place_forget()
# button_snr.place(
#     x=325.1809387207031,
#     y=260.1689147949219,
#     width=267.0,
#     height=41.0
# )


button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_AB = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: corr1Button(medicion),
    relief="flat"
)
button_AB.place_forget()
# button_AB.place(
#     x=325.1809387207031,
#     y=341.6893005371094,
#     width=267.0,
#     height=41.0
# )

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_BA = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: corr2Button(medicion, measurement_id),
    relief="flat"
)
button_AB.place_forget()
# button_BA.place(
#     x=325.1809387207031,
#     y=391.79034423828125,
#     width=267.0,
#     height=41.0
# )

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_ver_corr = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: verCorrButton(medicion, measurement_id),
    relief="flat"
)
button_ver_corr.place_forget()
# button_ver_corr.place(
#     x=661.0,
#     y=368.0,
#     width=288.1216125488281,
#     height=38.0
# )

button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_repetir = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: repetirButton(),
    relief="flat"
)
button_repetir.place_forget()
# button_repetir.place(
#     x=974.6044921875,
#     y=342.23681640625,
#     width=186.16131591796875,
#     height=38.0
# )

button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_aceptar = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: aceptarButton(),
    relief="flat"
)
button_aceptar.place_forget()
# button_aceptar.place(
#     x=974.1980590820312,
#     y=392.3378601074219,
#     width=189.03424072265625,
#     height=38.0
# )

button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_absorption = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: absorptionButton(medicion),
    relief="flat"
)
button_absorption.place_forget()
# button_absorption.place(
#     x=325.0,
#     y=511.0,
#     width=267.0,
#     height=63.687782287597656
# )

button_image_10 = PhotoImage(
    file=relative_to_assets("button_10.png"))
button_results = Button(
    image=button_image_10,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: resultsButton(medicion),
    relief="flat"
)
button_results.place_forget()
# button_results.place(
#     x=661.0,
#     y=496.0,
#     width=288.1216125488281,
#     height=38.0
# )

button_image_11 = PhotoImage(
    file=relative_to_assets("button_11.png"))
button_export = Button(
    image=button_image_11,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: exportButton(medicion),
    relief="flat"
)
button_export.place_forget()
# button_export.place(
#     x=829.0,
#     y=600.0,
#     width=352.59637451171875,
#     height=74.72699737548828
# )

def infoButton():
    # Toplevel object which will
    # be treated as a new window
    helpWindow = Toplevel(window)

    # Sets the title of the Toplevel widget
    helpWindow.title("Más información")
    helpWindow.resizable(False, False)
    # sets the geometry of toplevel
    helpWindow.geometry("400x400")

    # A Label widget to show in toplevel
    Label(helpWindow,
          text=open('Información.txt', 'r', encoding='utf-8').read(), wraplength=250, justify="left").pack()


def measureButton():
    # 1 - Desde la GUI se inicializa una nueva medición y se definen algunas variables/constantes

    folder_name = entry_people.get()
    speaker_mic_distance = 0.283
    mic_distance = 75e-3
    sound_speed = 343.2 * np.sqrt((273.15 + float(entry_temperature.get()))/293)             # Cálculo de velocidad desde la temperatura
    air_density = 1.18
    sample_name = entry_sample.get()
    sample_description = ' '

    global medicion
    # 2 - Se instancia un objeto Absorcion_10534_2

    medicion = Absorcion_10534_2(folder_name=folder_name,
                                 speaker_mic_distance=speaker_mic_distance,
                                 mic_distance=mic_distance,
                                 sound_speed=sound_speed,
                                 air_density=air_density,
                                 sample_name=sample_name,
                                 sample_description=sample_description)

    carpeta_medicion = medicion.folder_path

    # 3 - Desde la GUI, el usuario elige el hardware y también configura otras variables/constantes
    lista_hardware = list(medicion.devices.keys())

    input_device_gui = 'Analogue 1 + 2 (Focusrite Usb A'  # 'MacBook Pro Microphone'
    output_device_gui = 'Altavoces (Focusrite Usb Audio)'  # 'MacBook Pro Speakers'

    buffer_size = 2048
    format = pyaudio.paInt16  # Formato de audio (16 bits PCM)
    channels_output = 1  # Número de canales (1 para mono, 2 para estéreo)
    channels_input = 2
    sample_rate = 44100  # Tasa de muestreo en Hz
    sweep_lower_frequency = 100
    sweep_upper_frequency = 2500
    sweep_duration = 30
    record_duration = sweep_duration
    # Alpha calculation constants
    smooth_response = True
    smooth_window_size = 100
    octave_fraction = 3
    global measurement_id
    measurement_id = 0


    medicion.measurement_setup(input_device_gui=input_device_gui,
                               output_device_gui=output_device_gui,
                               buffer_size=buffer_size,
                               format=format,
                               channels_output=channels_output,
                               channels_input=channels_input,
                               sample_rate=sample_rate,
                               sweep_lower_frequency=sweep_lower_frequency,
                               sweep_upper_frequency=sweep_upper_frequency,
                               sweep_duration=sweep_duration,
                               record_duration=record_duration,
                               smooth_response=smooth_response,
                               smooth_window_size=smooth_window_size,
                               octave_fraction=octave_fraction)

    # 4 - Se genera el sweep

    medicion.generate_sweep()

    # Actualización elementos GUI
    button_snr.place(
        x=325.1809387207031,
        y=260.1689147949219,
        width=267.0,
        height=41.0
    )
    canvas.itemconfig(text_snr, state="normal")


def snrButton(medicion):
    # 5 - Se mide relación señal a ruido

    global snr_id
    snr_id = 1  # Este id debe ir aumentando si la medición se repite, para que quede registro
    medicion.measure_snr(snr_id)

    medicion.snr_data["freq_noise_A"]
    medicion.snr_data["noise_frac_A"]
    medicion.snr_data["noise_frac_B"]
    medicion.snr_data["signal_frac_A"]
    medicion.snr_data["signal_frac_B"]
    medicion.snr_data["snr_A"]
    medicion.snr_data["snr_B"]

    canvas.itemconfig(text_result, state="normal")

    if np.all(medicion.snr_data["snr_A"][7:19] > 10) and np.all(medicion.snr_data["snr_B"][7:19] > 10):
        canvas.itemconfig(text_snr_results, text = "Canal 1 OK. \nCanal 2 OK.", fill = "green")

        canvas.itemconfig(text_correction, state="normal")      # Visibilidad de elementos ocultos

        button_AB.place(
            x=325.1809387207031,
            y=341.6893005371094,
            width=267.0,
            height=41.0
        )


    elif np.all(medicion.snr_data["snr_A"][7:19] > 10) and np.any(medicion.snr_data["snr_B"][7:19] < 10):
        canvas.itemconfig(text_snr_results, text = "Canal 1 OK. \nCanal 2 NO CUMPLE, ajuste el nivel de entrada o salida.", fill="orange")

    elif np.any(medicion.snr_data["snr_A"][7:19] < 10) and np.all(medicion.snr_data["snr_B"][7:19] > 10):
        canvas.itemconfig(text_snr_results, text = "Canal 1 NO CUMPLE, ajuste el nivel de entrada o salida. \nCanal 2 OK.", fill="orange")

    else:
        canvas.itemconfig(text_snr_results, text = "Canal 1 NO CUMPLE. Canal 2 NO CUMPLE. \nAjuste los niveles de entrada o salida.", fill="red")



def corr1Button(medicion):
    # 6 - Se mide la Calibración AB - BA
    global measurement_id
    measurement_id += 1
    medicion.measure_calibration(measurement_id, "AB")

    button_BA.place(                        # Visibilidad de elementos ocultos
        x=325.1809387207031,
        y=391.79034423828125,
        width=267.0,
        height=41.0
    )


def corr2Button(medicion, measurement_id):
    # 6 - Se mide la Calibración AB - BA
    medicion.measure_calibration(measurement_id, "BA")

    button_ver_corr.place(                  # Visibilidad de elementos ocultos
        x=661.0,
        y=368.0,
        width=288.1216125488281,
        height=38.0
    )


def verCorrButton(medicion, measurement_id):
    # Se calcula la calibración y se verifica si todo está ok visualmente, sino repetir el proceso
    medicion.calibration_calculation(measurement_id)

    frequency = medicion.correction_data["frequency"]
    H1_corr_1 = medicion.correction_data["H1_corr_1"]
    H1_corr_2 = medicion.correction_data["H1_corr_2"]
    Hc = medicion.correction_data["Hc"]
    Hc_smooth = medicion.correction_data["Hc_smooth"]

    fig, ax = plt.subplots(2, 1)
    fig.suptitle('Función de corrección')
    ax[0].plot(frequency, 20*np.log10(abs(Hc_smooth)), label = 'Magnitud Hc')
    ax[0].set(xlabel = 'Frecuencia [Hz]', ylabel='Magnitud [dBre]')
    ax[0].set_xlim([230,2094])
    ax[0].grid()

    ax[1].plot(frequency, np.angle(Hc_smooth)*180/np.pi, label = 'Fase Hc')
    ax[1].set(xlabel = 'Frecuencia [Hz]', ylabel='Fase [°]')
    ax[1].set_xlim([230, 2094])
    ax[1].grid()

    button_repetir.place(                               # Visibilidad de elementos ocultos
        x=974.6044921875,
        y=342.23681640625,
        width=186.16131591796875,
        height=38.0
    )
    button_aceptar.place(
        x=974.1980590820312,
        y=392.3378601074219,
        width=189.03424072265625,
        height=38.0
    )

    plt.show()


def repetirButton():
    print('Hay que hacer algo con los ID')

    # measurement_id = measurement_id + 1
    button_aceptar.place_forget()
    button_ver_corr.place_forget()
    button_BA.place_forget()
    button_repetir.place_forget()


def aceptarButton():
    canvas.itemconfig(text_absorption, state="normal")  # Visibilidad de elementos ocultos

    canvas.itemconfig(text_repetition, state="normal")

    entry_repetitions.place(
        x=431.89453125,
        y=466.0,
        width=83.53125,
        height=28.02450942993164
    )

    button_absorption.place(
        x=325.0,
        y=511.0,
        width=267.0,
        height=63.687782287597656
    )


def absorptionButton(medicion):
    n_rep = float(entry_repetitions.get())
    # abs_ids = [1, 2]
    abs_ids = np.arange(1, n_rep+1, 1)
    for id in abs_ids:
        medicion.measure_absorption(id)
        medicion.absorption_coefficient_calculation(id)


    button_results.place(                              # Visibilidad de elementos ocultos
        x=661.0,
        y=496.0,
        width=288.1216125488281,
        height=38.0
    )


def resultsButton(medicion):
    n_rep = float(entry_repetitions.get())
    # abs_ids = [1, 2]
    abs_ids = np.arange(1, n_rep+1, 1)

    frecuencia = medicion.absorption_data[1]["frequency"]
    alpha = medicion.absorption_data[1]["alpha_coeff"]
    r = medicion.absorption_data[1]["r_coeff"]
    z = medicion.absorption_data[1]["z_coeff"]
    frec_center = medicion.absorption_data[1]["freq_center"]
    alpha_center = medicion.absorption_data[1]["alpha_frac"]
    tf_corr = medicion.absorption_data[1]["tf"]

    for i in abs_ids:
        plt.plot(frec_center,medicion.absorption_data[i]["alpha_frac"], label = entry_sample.get() + '_' + str(i))
    # plt.plot(frec_center, alpha_center, label = entry_sample.get())
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Coeficiente de absorción sonora')
    plt.xlim([230,2094])
    plt.ylim([0,1])
    plt.legend()
    plt.grid()

    button_export.place(                               # Visibilidad de elementos ocultos
        x=829.0,
        y=600.0,
        width=352.59637451171875,
        height=74.72699737548828
    )

    entry_comments.place(
        x=85.0,
        y=622.0,
        width=675.0,
        height=51.0
    )

    canvas.itemconfig(text_comments, state="normal")

    plt.show()

def exportButton(medicion):
    N = int(entry_repetitions.get())
    frec_center = medicion.absorption_data[1]["freq_center"]
    alpha_center = medicion.absorption_data[1]["alpha_frac"]

    data_excel = {'Frecuencia 1/3 oct [Hz]': medicion.absorption_data[1]["freq_center"]}
    df = pd.DataFrame(data_excel)
    data_alpha = {"Alpha_" + str(i + 1): medicion.absorption_data[i]["alpha_frac"] for i in range (1,N+1)}

    index_columna_frecuencia = df.columns.get_loc('Frecuencia 1/3 oct [Hz]')
    for nombre_columna, datos_columna in data_alpha.items():
        df.insert(index_columna_frecuencia + 1, nombre_columna, datos_columna)


    # data_excel = {'Frecuencia lin': frequency, 'Alpha 0': alpha[0], 'Alpha 1': alpha[1], 'Alpha 2': alpha[2]}
    # df = pd.DataFrame(datos)
    # excelName = os.path.join(medicion.folder_name, '-Resultados.xlsx')
    excelName = medicion.folder_path + '/Resultados.xlsx'
        # 'C:/Users/MAITE/Documents/UNTREF/Tesis/Procesamiento/GitHub/Tesis/audios-test/ProtoTubo/23-11-23/Acustiver R/Acustiver R #3. 23-11 TERCIOS.xlsx'
    df.to_excel(excelName, index=False)


def botones():
    print('entry1' + entry_comments.get())
    print('entry2' + entry_repetitions.get())
    print('entry3' + entry_temperature.get())
    print('entry4' + entry_sample.get())
    print('entry5' + entry_people.get())


window.resizable(False, False)
window.mainloop()
