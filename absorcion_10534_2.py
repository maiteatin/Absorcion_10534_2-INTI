import os
from datetime import datetime
import sounddevice as sd
import numpy as np
import pyaudio
import wave
import time
import threading
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from scipy.fft import fft, ifft, fftfreq, rfft
import matplotlib.pyplot as plt



class Absorcion_10534_2:
    def __init__(self, folder_name, speaker_mic_distance, mic_distance, sound_speed, air_density, sample_name, sample_description):
        """
        Variables de entrada:
            - folder_name: Nombre de la carpeta para almacenar los resultados.
            - mic_distance: Distancia entre micrófonos.
            - sound_speed: Velocidad del sonido en el aire.
            - air_density: Densidad del aire.
            - sample_name: Nombre del ejemplo de medición.
            - sample_description: Descripción del ejemplo de medición.
        Variables de salida: Ninguna explícita.
        Función: Inicializa la clase y genera la carpeta para almacenar resultados.
        """

        self.folder_name = folder_name
        self.speaker_mic_distance = speaker_mic_distance
        self.mic_distance = mic_distance
        self.sound_speed = sound_speed
        self.air_density = air_density
        self.sample_name = sample_name
        self.sample_description = sample_description
        self.measurements = {
            "signal_to_noise_check" : {},
            "calibration_AB" : {},
            "calibration_BA" : {},
            "absorption": {}
        }
        self.absorption_data = {}

        self._generar_carpeta()
        self._list_sounddevices()

    def _generar_carpeta(self):
        """
        Variables de entrada: Ninguna explícita.
        Variables de salida: Ninguna explícita.
        Función: Genera una carpeta con un nombre específico basado en la fecha y hora actuales.
        """
        # Obtener la fecha y hora actual
        ahora = datetime.now()

        # Formatear la fecha y hora en el formato deseado
        formato_fecha_hora = ahora.strftime("%d_%M_%Y_%H_%M")

        # Pedir al usuario que ingrese un nombre para la carpeta
        nombre_usuario = self.folder_name
        nombre_muestra = self.sample_name

        # Crear el nombre completo de la carpeta combinando el formato y el nombre del usuario
        nombre_carpeta = f"{formato_fecha_hora}_{nombre_usuario}_{nombre_muestra}"

        # Crear la carpeta si no existe
        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)
            print(f"Carpeta '{nombre_carpeta}' creada con éxito.")
        else:
            print(f"La carpeta '{nombre_carpeta}' ya existe.")

        self.folder_path = os.path.abspath(nombre_carpeta)

    def _list_sounddevices(self):
        """
        Variables de entrada: Ninguna explícita.
        Variables de salida: Ninguna explícita.
        Función: Lista los dispositivos de audio ASIO disponibles.
        """

        # Obtener una lista de dispositivos de audio ASIO
        asio_devices = sd.query_devices()
        # pprint.pprint(list(asio_devices))
        if not asio_devices:
            print("No se encontraron dispositivos ASIO disponibles.")
        else:
            print("Dispositivos ASIO disponibles:")

            # for idx, device in enumerate(asio_devices):

        devices = {}
        for device in asio_devices:
            devices.update(
                {device["name"]: device}
            )
        self.devices = devices

    def measurement_setup(self, input_device_gui, output_device_gui,
                          buffer_size, format, channels_output,
                          channels_input, sample_rate,
                          sweep_lower_frequency, sweep_upper_frequency,
                          sweep_duration, record_duration,
                          smooth_response = True, smooth_window_size = 5,
                          octave_fraction = 3):
        """
        Variables de entrada:
            - input_device_gui: Dispositivo de entrada seleccionado por el usuario.
            - output_device_gui: Dispositivo de salida seleccionado por el usuario.
            - buffer_size: Tamaño del búfer de audio.
            - format: Formato de audio.
            - channels_output: Número de canales de salida.
            - channels_input: Número de canales de entrada.
            - sample_rate: Tasa de muestreo.
            - sweep_lower_frequency: Frecuencia inferior del barrido.
            - sweep_upper_frequency: Frecuencia superior del barrido.
            - sweep_duration: Duración del barrido.
            - record_duration: Duración de la grabación.
        Variables de salida: Ninguna explícita.
        Función: Configura los parámetros para la medición.
        """
        
        self.input_device = input_device_gui
        self.output_device = output_device_gui
        self.buffer_size = buffer_size
        self.format = format
        self.channels_output = channels_output
        self.channels_input = channels_input
        self.sample_rate = sample_rate
        self.sweep_lower_frequency = sweep_lower_frequency
        self.sweep_upper_frequency = sweep_upper_frequency
        self.sweep_duration = sweep_duration
        self.record_duration = record_duration
        self.smooth_response = smooth_response
        self.smooth_window_size = smooth_window_size
        self.octave_fraction = octave_fraction
  

    def generate_sweep(self):
        """Function generates a logarithmic sine sweep and inverse filter between the specified input frequencies
        and sweep duration. Convolution between the sweep and its inverse gives IR response.
        INPUTS:
            f1: initial frequency of sweep [Hz]
            f2: final frequency of sweep [Hz]
            time: sweep length [s]
            fs: sample rate [Hz] """
        from scipy.io.wavfile import write

        f1 = self.sweep_lower_frequency
        f2 = self.sweep_upper_frequency
        time = self.sweep_duration
        fs = self.sample_rate

        t = np.arange(0, round(time), 1 / fs)
        w1 = 2 * np.pi * f1
        w2 = 2 * np.pi * f2
        K = (w1 / np.log(w2 / w1)) * time
        L = time / np.log(w2 / w1)
        y = np.sin(K * (np.exp(t / L) - 1))  # Log Sine Sweep

        w = (K / L) * np.exp(t / L)
        m = w1 / w

        u = m * np.flip(y)
        u = u / max(abs(u))  # Inverse Log Sine Sweep

        nombre_archivo = f'Sweep_{f1}_{f2}_{time}_{fs}.wav'
        ruta_completa = os.path.join(self.folder_path, nombre_archivo)
        y = (y * 32767).astype(np.int16)  # Convertir a tipo entero de 16 bits
        write(ruta_completa, self.sample_rate, y)
        self.sweep_location = ruta_completa

        nombre_archivo = f'InverseFilter_{f1}_{f2}_{time}_{fs}.wav'
        ruta_completa = os.path.join(self.folder_path, nombre_archivo)
        u = (u * 32767).astype(np.int16)  # Convertir a tipo entero de 16 bits
        write(ruta_completa, self.sample_rate, u)
        self.inverse_filter_location = ruta_completa


    def measure_snr(self,measurement_id):
        """
        Variables de entrada:
            - measurement_id: Identificador de la medición.
        Variables de salida: Ninguna explícita.
        Función: Realiza la medición de la relación señal a ruido (SNR).
        """

        # Medir Relación señal a Ruido
        # Pasos:
            # Agregar el registro
            # Grabar Ruido de fondo 
            # Grabar señal 
            # Actualizar registro
            # Calcular snr y retornar valores 

        # Agregar al registro
        self._add_measurement("signal_to_noise_check",measurement_id)

        nombre_archivo_ruido = os.path.join(self.folder_path, self.measurements["signal_to_noise_check"][measurement_id]["audio_files"]["noise"])

        # Grabar ruido de fondo 
        try:
            self._record_audio(
                output_filename=os.path.join(self.folder_path, nombre_archivo_ruido),
                record_duration=self.record_duration,
                measurement_type = "Ruido de fondo para relación señal a ruido"
            )
        except Exception as e:
            print(f"Error al grabar audio, {e}")
        
        # Grabar señal 
        measurement_type = "Señal para relación señal a ruido"
        nombre_archivo_señal = os.path.join(self.folder_path, self.measurements["signal_to_noise_check"][measurement_id]["audio_files"]["signal"])

        try:
            thread = threading.Thread(target=self._play_and_record_audio,args=(nombre_archivo_señal,measurement_type))
            thread.start()
            thread.join()
        except Exception as e:
            print(f"Error al grabar audio, {e}")

        # Actualizar registro
        self.measurements["signal_to_noise_check"][measurement_id]["status"] = "recorded"

        # Calcular snr

        señal = self._read_audio(audio_path=nombre_archivo_señal)
        # señal = self._read_audio('C:/Users/MAITE/Documents/UNTREF/Tesis/Procesamiento/GitHub/Tesis/build/12_16_2023_10_16_Maite_Acustiver R - INTI 3/signal_to_noise_check_1_signal.wav')
        ruido = self._read_audio(audio_path=nombre_archivo_ruido)
        # ruido = self._read_audio('C:/Users/MAITE/Documents/UNTREF/Tesis/Procesamiento/GitHub/Tesis/build/12_16_2023_10_16_Maite_Acustiver R - INTI 3/signal_to_noise_check_1_noise.wav')

        freq_noise_A, noise_frac_A, noise_frac_B, signal_frac_A, signal_frac_B, snr_A, snr_B = self._snr_calculation(ruido, señal)

        # Actualizar registro
        self.measurements["signal_to_noise_check"][measurement_id]["status"] = "calculo_completo"

        return freq_noise_A, noise_frac_A, noise_frac_B, signal_frac_A, signal_frac_B, snr_A, snr_B

    
    def measure_calibration(self,measurement_id, mic_setup):
        """
        Variables de entrada:
            - measurement_id: Identificador de la medición.
            - mic_setup: Configuración de micrófonos ("AB" o "BA").
        Variables de salida: Ninguna explícita.
        Función: Realiza la medición de calibración para la configuración de micrófonos dada.
        """
        
        # Pasos:
            # Agregar el registro
            # Grabar señal posición dada
            # Actualizar registro
      
        if mic_setup == "AB":
            # Agregar al registro
            self._add_measurement("calibration_AB",measurement_id)

            # Tomo nombre de archivo
            nombre_archivo = os.path.join(self.folder_path, self.measurements["calibration_AB"][measurement_id]["audio_files"]["calibration_AB"])
            
            measurement_type = "calibration_AB"
            # Grabar señal en posicion dada
            try:
                thread = threading.Thread(target=self._play_and_record_audio,args=(nombre_archivo,measurement_type))
                thread.start()
                thread.join()
            except Exception as e:
                print(f"Error al grabar audio, {e}")
            
            # Actualizar registro
            self.measurements["calibration_AB"][measurement_id]["status"] = "recorded"

        elif mic_setup == "BA":
            # Agregar al registro
            self._add_measurement("calibration_BA",measurement_id)

            # Tomo nombre de archivo
            # Tomo nombre de archivo
            nombre_archivo = os.path.join(self.folder_path, self.measurements["calibration_BA"][measurement_id]["audio_files"]["calibration_BA"])
            measurement_type = "Calibración BA"
            # Grabar señal en posicion dada
            try:
                thread = threading.Thread(target=self._play_and_record_audio,args=(nombre_archivo,measurement_type))
                thread.start()
                thread.join()
            except Exception as e:
                print(f"Error al grabar audio, {e}")
            
            # Actualizar registro
            self.measurements["calibration_BA"][measurement_id]["status"] = "recorded"


    def calibration_calculation(self, measurement_id):
        # Check if measurement is completed
        if (self.measurements["calibration_BA"][measurement_id]["status"] == "recorded") & (self.measurements["calibration_AB"][measurement_id]["status"] == "recorded"):
            
            audio_file_AB = os.path.join(self.folder_path,self.measurements["calibration_AB"][measurement_id]["audio_files"]["calibration_AB"])
            audio_file_BA = os.path.join(self.folder_path,self.measurements["calibration_BA"][measurement_id]["audio_files"]["calibration_BA"]) 

            # CORRECTION
            audio_corr_1 = self._read_audio(audio_file_AB)   # Raw audio read original position from its path
            audio_corr_2 = self._read_audio(audio_file_BA)

            # audio_corr_1 = self._read_audio('C:/Users/MAITE/Documents/UNTREF/Tesis/Procesamiento/GitHub/Tesis/0711 - P3 AB -1.wav')   # Raw audio read original position from its path
            # audio_corr_2 = self._read_audio('C:/Users/MAITE/Documents/UNTREF/Tesis/Procesamiento/GitHub/Tesis/0711 - P3 BA -1.wav')

            # Original microphones position
            _, H1_corr_1, _, _, _, frequency = self._transferFunction(audio_corr_1[:,0],audio_corr_1[:,1])
            # Switched microphones position
            _, H1_corr_2, _, _, _, _ = self._transferFunction(audio_corr_2[:,0],audio_corr_2[:,1])

            # Correction transfer function from two measurement samples
            Hc = np.sqrt(H1_corr_1*H1_corr_2)       

            # Check if we have to apply a median filter to smooth the response
            if self.smooth_response == True:
                size = self.smooth_window_size
                Hc_mg_smooth = median_filter(abs(Hc), size)
                Hc_ph_smooth = median_filter(np.angle(Hc), size)
                Hc_smooth = Hc_mg_smooth * np.exp(1j * Hc_ph_smooth)
            else:
                Hc_smooth = Hc

            correction = {
                "frequency":frequency,
                "H1_corr_1":H1_corr_1,
                "H1_corr_2":H1_corr_1,
                "Hc":Hc,
                "Hc_smooth":Hc_smooth
            }

            self.correction_data = correction

            return "Corrección calculada" 
        else:
            return "Medición de correción incompleta" 
       


    def measure_absorption(self,measurement_id):
        """
        Variables de entrada:
            - measurement_id: Identificador de la medición.
        Variables de salida: Ninguna explícita.
        Función: Realiza la medición de absorción.
        """
        # Medir absorcion
        # Pasos:
            # Agregar el registro
            # Grabar transferencia 
            # Actualizar registro

        # Agregar al registro
        self._add_measurement("absorption",measurement_id)
        
        nombre_archivo = os.path.join(self.folder_path, self.measurements["absorption"][measurement_id]["audio_files"]["transfer"])
        measurement_type = "transferencia para Coeficiente de Absorción"
        # Grabar señal en posicion dada
        try:
            thread = threading.Thread(target=self._play_and_record_audio,args=(nombre_archivo,measurement_type))
            thread.start()
            thread.join()
        except Exception as e:
            print(f"Error al grabar audio, {e}")

        # Actualizar registro
        self.measurements["absorption"][measurement_id]["status"] = "recorded"

        
    def _add_measurement(self, measurement_type, measurement_id):
        """
        Variables de entrada:
            - measurement_type: Tipo de medición ("signal_to_noise_check", "absorption", "calibration_AB", "calibration_BA").
            - measurement_id: Identificador de la medición.
        Variables de salida: Ninguna explícita.
        Función: Agrega una nueva medición al registro.
        """

        if measurement_type == "signal_to_noise_check":

            self.measurements[measurement_type].update({
                    measurement_id:{
                    "audio_files": {
                        "signal":measurement_type + "_" + str(measurement_id) + "_signal.wav",
                        "noise":measurement_type + "_" + str(measurement_id) + "_noise.wav",
                        },

                    "status": "started"
                    }       
                })
        elif measurement_type == "absorption":
            self.measurements[measurement_type].update({
                    measurement_id: {
                    "audio_files": {
                        "transfer":measurement_type + "_" + str(measurement_id) + "_transfer.wav",
                        },
                    "status": "started"
                    }
                })
        elif measurement_type == "calibration_AB":
            self.measurements[measurement_type].update({
                    measurement_id: {
                    "audio_files": {
                        "calibration_AB":measurement_type + "_" + str(measurement_id) + "_calibration_AB.wav"
                        },
                    "status": "started"
                    }
                })
        elif measurement_type == "calibration_BA":
            self.measurements[measurement_type].update({
                    measurement_id:{
                    "audio_files": {
                        "calibration_BA":measurement_type + "_" + str(measurement_id) + "_calibration_BA.wav"
                        },
                    "status": "started"
                    }
                })
        else:
            raise  ValueError(f"El tipo de medición debe ser uno de: '{self.measurements.keys()}'.")


    def absorption_coefficient_calculation(self,measurement_id):

        # stereo_audio_file = self.measurements["absorption"][measurement_id]["audio_files"]["transfer"]
        stereo_audio_file = os.path.join(self.folder_path, self.measurements["absorption"][measurement_id]["audio_files"]["transfer"])

        # TRANSFER FUNCTION
        # Read stereo audio file to numpy array
        data = self._read_audio(stereo_audio_file)
        # data = self._read_audio('C:/Users/MAITE/Documents/UNTREF/Tesis/Procesamiento/GitHub/Tesis/0711 - P3 AB -1.wav')
        # Calculate transfer function
        _, H1, _, _, _, frequency = self._transferFunction(data[:,0],data[:,1])

        # Check if we have to apply a median filter to smooth the response
        if self.smooth_response == True:
            size = self.smooth_window_size
            mg_smooth = median_filter(abs(H1), size)            # Magnitude
            ph_smooth = median_filter(np.angle(H1), size)       # Phase
            tf_smooth = mg_smooth * np.exp(1j * ph_smooth)       # Complex smoothed TF
            Hc = self.correction_data["Hc_smooth"]      
        else:
            tf_smooth = H1    
            Hc = self.correction_data["Hc"]                                                                                    

        # Corrected transfer function
        tf_corr = tf_smooth / Hc

        # COEFFICIENTS CALCULATION
        alpha_coeff, r_coeff, z_coeff = self._alpha_calculation(frequency, tf_corr, self.speaker_mic_distance, self.mic_distance, self.sound_speed)
        
        
        # Llamar al alpha por bandas de octaba o tercios
        freq_center, alpha_frac = self._fractional_bands(alpha_coeff, frequency, self.octave_fraction)

        absorption_data = {
            "frequency" : frequency,
            "alpha_coeff" : alpha_coeff,
            "r_coeff" : r_coeff,
            "z_coeff" : z_coeff,
            "freq_center" : freq_center,
            "alpha_frac" : alpha_frac,
            "tf" : tf_corr
        }

        # self.absorption_data = absorption_data

        self.absorption_data.update({measurement_id: absorption_data})


    def _transferFunction(self,data1, data2):
        """ Processing generates transfer functions (TF) between two signals (data1, data2) with the same sample frequency.
        It returns TF obtained from the cross-spectrum and auto-spectrum of data1 and data 2 (H1 and H2 respectively)
        and the geometric mean of these (H) """

        samplerate = self.sample_rate

        chA_fft = np.fft.rfft(data1)                        # Channel A FFT calculation
        chB_fft = np.fft.rfft(data2)                        # Channel B FFT calculation

        frequency = np.fft.rfftfreq(len(data1), 1/samplerate)    # FFT frequencies

        G_AA = chA_fft * np.conjugate(chA_fft)              # Channel A auto-spectrum
        G_BB = chB_fft * np.conjugate(chB_fft)              # Channel B auto-spectrum
        G_AB = chB_fft * np.conjugate(chA_fft)              # Cross-spectrum between channel A and B
        G_AB_conj = chA_fft * np.conjugate(chB_fft)

        H1 = G_AB / G_AA                                    # Transfer function
        H2 = G_BB / G_AB_conj                               # Transfer function

        H = np.sqrt(H1*H2)                                  # Transfer function

        return H, H1, H2, chA_fft, chB_fft,frequency

    def _record_audio(self,output_filename,record_duration,measurement_type):
        """
        Variables de entrada:
            - measurement_type: Tipo de medición ("signal_to_noise_check", "absorption", "calibration_AB", "calibration_BA").
            - measurement_id: Identificador de la medición.
        Variables de salida: Ninguna explícita.
        Función: Agrega una nueva medición al registro.
        """
        
        device_name = self.input_device
        buffer_size = self.buffer_size 
        format = self.format 
        channels = self.channels_input 
        sampling_rate = self.sample_rate
    
        # Crea un objeto PyAudio
        p = pyaudio.PyAudio()

        # Busca el dispositivo específico por nombre
        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["name"] == device_name:
                device_index = i
                break

        if device_index is None:
            raise ValueError(f"No se encontró el dispositivo '{device_name}'.")

        # Configura los parámetros de grabación con el dispositivo específico
        stream = p.open(
            format=format,
            channels=channels,
            rate=sampling_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=buffer_size,
        )

        print(f"Grabando {measurement_type}...")

        frames = []

        # Inicia la grabación
        for _ in range(0, int(sampling_rate / buffer_size * record_duration)):
            data = stream.read(buffer_size)
            frames.append(data)

        # Detiene la grabación
        print("Grabación finalizada.")
        stream.stop_stream()
        stream.close()

        # Cierra el objeto PyAudio
        p.terminate()

        # Guarda la grabación en un archivo WAV
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(sampling_rate)
            wf.writeframes(b''.join(frames))

        print(f"La grabación se ha guardado en '{output_filename}'.")

    def _play_and_record_audio(self,output_filename,measurement_type):
        """
        Variables de entrada:
            - output_filename: Nombre del archivo de salida.
        Variables de salida: Ninguna explícita.
        Función: Reproduce un barrido de señal y graba la respuesta utilizando un hilo separado.
        """
    
        delta_grabacion = 5
        
        device_name_input = self.input_device
        input_channels = self.channels_input 

        device_name_output = self.output_device
        output_channels = self.channels_output

        buffer_size = self.buffer_size 
        format = self.format 
        sampling_rate = self.sample_rate

        input_filename = self.sweep_location


        # Índice del dispositivo de salida que deseas utilizar
        #device_name_input = "Analogue 1 + 2 (Focusrite Usb A"
        #device_name_output = 'Altavoces (Focusrite Usb Audio)'

        p = pyaudio.PyAudio()
        # Busca el dispositivo específico por nombre
        input_device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["name"] == device_name_input:
                input_device_index = i
                break

        if input_device_index is None:
            raise ValueError(f"No se encontró el dispositivo '{device_name_input}'.")

        output_device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["name"] == device_name_output:
                output_device_index = i
                break

        if output_device_index is None:
            raise ValueError(f"No se encontró el dispositivo '{device_name_output}'.")
        
        # Stream para la reproducción desde el archivo
        input_wave = wave.open(input_filename, 'rb')
        print(input_wave.getframerate())
        play_stream = p.open(format=p.get_format_from_width(input_wave.getsampwidth()),
                            channels=output_channels,
                            rate=sampling_rate,
                            output=True,
                            output_device_index=output_device_index)

        # Stream para la grabación
        record_stream = p.open(format=format,
                            channels=input_channels,
                            rate=sampling_rate,
                            input=True,
                            input_device_index=input_device_index,
                            frames_per_buffer=buffer_size)

        print(f"Reproduciendo sweep y grabando {measurement_type}")

        # Listas para almacenar los bloques de audio grabados
        frames = []
        inicio_tiempo = 0
        while True:
            # Leer datos del stream de grabación
            input_data = record_stream.read(buffer_size)
            frames.append(input_data)

            # Reproducir datos en el stream de reproducción desde el archivo
            play_data = input_wave.readframes(buffer_size)
            if (not play_data) and (inicio_tiempo == 0):
                inicio_tiempo = time.time()
            elif (not play_data) and (time.time() - inicio_tiempo >= delta_grabacion):
                break
            play_stream.write(play_data)

        # Detener y cerrar los streams
        play_stream.stop_stream()
        play_stream.close()
        record_stream.stop_stream()
        record_stream.close()
        p.terminate()

        # Guardar la grabación en un archivo WAV
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(input_channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(sampling_rate)
            wf.writeframes(b''.join(frames))

        print(f"La grabación se ha guardado en '{output_filename}'.")

    def _read_audio(self, audio_path):
        """Reads an audio file from its location (path)
            Formats supported: .wav

            Returns the following information:
            data: audio data
            path: File path, name and its extension
            duration: duration of the file in seconds
            samplerate: sampling frequency in Hz
            frames: number of samples
            channels: 1 (mono) or 2 (stereo)"""
        import soundfile as sf

        data, samplerate = sf.read(audio_path)
        frames = data.shape[0]
        channels = len(data.shape)
        duration = 1 / samplerate * frames

        # self.audio_data = data

        return data

    def _fractional_bands(self, data, frequency, fraction):
        """Filtra en fracciones de banda de octava según sea "fraction" un espectro determinado (alpha o SPL)"""

        def find_nearest(array, value):  # Find the corresponding index to the nearest 'value' in an 'array'
            idx = (np.abs(array - value)).argmin()
            return idx

        frac = fraction
        f_center_1 = np.array([31.5, 63, 125, 250, 500, 1000, 2000])
        fl_1 = f_center_1 * 2 ** (-1 / 2)
        fu_1 = f_center_1 * 2 ** (1 / 2)
        f_center_3 = np.array(
            [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000])
        fl_3 = f_center_3 * 2 ** (-1 / 6)
        fu_3 = f_center_3 * 2 ** (1 / 6)
        #frac = 3

        if frac == 1:
            data_frac = np.zeros(len(f_center_1), dtype=np.float64)
            freq_center = f_center_1
            for i in range(len(f_center_1)):
                idx_l = find_nearest(frequency, fl_1[i])
                idx_u = find_nearest(frequency, fu_1[i])
                data_frac[i] = (np.sum(data[idx_l:idx_u]) / (idx_u - idx_l))
        else:
            data_frac = np.zeros(len(f_center_3), dtype=np.float64)
            freq_center = f_center_3
            for i in range(len(f_center_3)):
                idx_l = find_nearest(frequency, fl_3[i])
                idx_u = find_nearest(frequency, fu_3[i])
                data_frac[i] = (np.sum(data[idx_l:idx_u]) / (idx_u - idx_l))

        return freq_center, data_frac

    def _fractional_bands_snr(self, data, frequency, fraction):
        """Filtra en fracciones de banda de octava según sea "fraction" un espectro determinado (alpha o SPL)"""
        def find_nearest(array, value):  # Find the corresponding index to the nearest 'value' in an 'array'
            idx = (np.abs(array - value)).argmin()
            return idx

        frac = fraction
        f_center_1 = np.array([31.5, 63, 125, 250, 500, 1000, 2000])
        fl_1 = f_center_1 * 2 ** (-1 / 2)
        fu_1 = f_center_1 * 2 ** (1 / 2)
        f_center_3 = np.array(
            [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000])
        fl_3 = f_center_3 * 2 ** (-1 / 6)
        fu_3 = f_center_3 * 2 ** (1 / 6)
        frac = 3

        if frac == 1:
            data_frac = np.zeros(len(f_center_1), dtype=np.float64)
            freq_center = f_center_1
            for i in range(len(f_center_1)):
                idx_l = find_nearest(frequency, fl_1[i])
                idx_u = find_nearest(frequency, fu_1[i])
                data_frac[i] = np.sum(np.square(abs(data[idx_l:idx_u])))
        else:
            data_frac = np.zeros(len(f_center_3), dtype=np.float64)
            freq_center = f_center_3
            for i in range(len(f_center_3)):
                idx_l = find_nearest(frequency, fl_3[i])
                idx_u = find_nearest(frequency, fu_3[i])
                data_frac[i] = np.sum(np.square(abs(data[idx_l:idx_u])))

        return freq_center, data_frac

    def _snr_calculation(self, noise, signal):

        sample_rate = self.sample_rate

        # noise_fft = np.fft.rfft(noise)  # Dual channel
        # signal_fft = np.fft.rfft(signal)  # Dual channel

        noise_fft_A = np.fft.rfft(noise[:, 0])  # Dual channel
        noise_fft_B = np.fft.rfft(noise[:, 1])
        signal_fft_A = np.fft.rfft(signal[:, 0])  # Dual channel
        signal_fft_B = np.fft.rfft(signal[:, 1])

        freq_snr = np.fft.rfftfreq(len(noise), 1 / sample_rate)

        freq_noise_A, noise_frac_A = self._fractional_bands_snr(noise_fft_A, freq_snr, 3)  # Fractional octave bands noise channel A
        freq_noise_B, noise_frac_B = self._fractional_bands_snr(noise_fft_B, freq_snr, 3)  # Fractional octave bands noise channel B

        freq_signal_A, signal_frac_A = self._fractional_bands_snr(signal_fft_A, freq_snr, 3)  # Fractional octave bands signal channel A
        freq_signal_B, signal_frac_B = self._fractional_bands_snr(signal_fft_B, freq_snr, 3)  # Fractional octave bands signal channel B

        # snr_A = 20 * np.log10(abs(signal_frac_A)) - 20 * np.log10(abs(noise_frac_A))
        # snr_B = 20 * np.log10(abs(signal_frac_B)) - 20 * np.log10(abs(noise_frac_B))

        snr_A = 10 * np.log10(signal_frac_A / noise_frac_A)
        snr_B = 10 * np.log10(signal_frac_B / noise_frac_B)

        snr_data = {
            "freq_noise_A":freq_noise_A, 
            "noise_frac_A":noise_frac_A, 
            "noise_frac_B":noise_frac_B, 
            "signal_frac_A":signal_frac_A, 
            "signal_frac_B":signal_frac_B, 
            "snr_A":snr_A, 
            "snr_B":snr_B
        }

        self.snr_data = snr_data

        return freq_noise_A, noise_frac_A, noise_frac_B, signal_frac_A, signal_frac_B, snr_A, snr_B

        
    def _alpha_calculation(self,freq, H, x1, s, c):
        k = 2 * np.pi * freq / c
        HI = np.ones_like(H) * np.exp(-1j * k * s)
        HR = np.ones_like(H) * np.exp(1j * k * s)

        #
        # # H = H1
        r = ((H - HI) / (HR - H)) * np.exp(2j * k * x1)
        alpha = abs(1 - abs(r ** 2))
        z = (1 + r)/(1 - r)

        return alpha, r, z

   