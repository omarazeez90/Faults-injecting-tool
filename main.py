import glob
import os
import subprocess
import PySimpleGUI as sg
from PIL import Image
from Faults import FaultsInjectionSystem

size = (12, 10)
sg.theme("DarkBlue3")
current_dir = os.getcwd()
faults_injecting = FaultsInjectionSystem()
output_folder = '\output//'
output_size = (224, 224)
# global strength_list
strength_list = []
source_path = current_dir
output_path = current_dir

folder_layout = [
    sg.Frame('Set the source samples folder and the output paths:', [
        [sg.Combo(sg.user_settings_get_entry('target_folder', []), default_value=current_dir, size=(59, 1),
                  key='set_outputpath'),
         sg.FolderBrowse(button_text='Output Path', initial_folder=current_dir)],
        [sg.Combo(sg.user_settings_get_entry('target_folder', []), default_value=current_dir, size=(59, 1),
                  key='set_sourcepath'),
         sg.FolderBrowse(button_text='Source Path', initial_folder=current_dir)]], key='COL', border_width=1, )]

strength_layout = [
    sg.Frame('Specify the strength of faults:', [[
        sg.Checkbox('strength 1', default=False, size=size, key="fault_strength1"),
        sg.Checkbox('strength 2', default=False, size=size, key="fault_strength2"),
        sg.Checkbox('strength 3', default=False, size=size, key="fault_strength3"),
        sg.Checkbox('strength 4', default=False, size=size, key="fault_strength4")], [
        sg.Checkbox('strength 5', default=False, size=size, key="fault_strength5")]], key='COL',
             border_width=1, )]

faults_layout = [
    sg.Frame('Select the desired faults:', [[
        sg.Checkbox('Condensation', default=False, size=size, key="condensation"),
        sg.Checkbox('Crack', default=False, size=size, key="crack"),
        sg.Checkbox('Dirt', default=False, size=size, key="dirt"),
        sg.Checkbox('Ice', default=False, size=size, key="ice")
    ], [
        sg.Checkbox('Rain', default=False, size=size, key="rain"),
        sg.Checkbox('Blur', default=False, size=size, key="blur"),
        sg.Checkbox('Brightness', default=False, size=size, key="brightness"),
        sg.Checkbox('Speckle noise', default=False, size=size, key="speckle_noise")
    ], [
        sg.Checkbox('Sharpness', default=False, size=size, key="sharpness"),
        sg.Checkbox('Dead pixels', default=False, size=size, key="dead_pixels"),
        sg.Checkbox('No Demosaicing', default=False, size=size, key="no_Demosaicing"),
        sg.Checkbox('Darkness', default=False, size=size, key="darkness")
    ], [
        sg.Checkbox('Dead pixels line', default=False, size=size, key="dead_pixels_line"),
        sg.Checkbox('No Bayer Filter', default=False, size=size, key="no_bayer_filter"),
        sg.Checkbox('No Chromatic Aberration Correction', default=False, size=(25, 10), key="no_chrom_ac")
    ]], key='COL', border_width=1, )]

done_button = [sg.Button('Start generating', font=('Times New Roman', 12), key="start"),
               sg.Button('Exit', font=('Times New Roman', 12), key="Exit")]

set_button = [sg.Radio('Generate all faults and strength', 'rd_all', default=False, key='all_fault'),
              sg.Radio('Generate selected', 'rd_all', default=False, key='selected_faults'), ]

progress_bar = [[sg.Text("Generating the faulty images in the output directory:", font='Lucida')],
                [sg.ProgressBar(20, size=(50, 5), border_width=4, key='progbar')]]
layout = folder_layout, strength_layout, faults_layout, done_button, progress_bar

# Create the window
window = sg.Window("Generate faulty frames", layout, margins=(40, 40))

# Create an event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        window.close()
        break
    if event == 'Output Path':
        sg.user_settings_set_entry('target_folder', [values['set_outputpath']])
    elif event == 'Source Path':
        sg.user_settings_set_entry('target_folder', [values['set_sourcepath']])

    if event == 'start':
        faults_injecting.condensation = values["condensation"]
        faults_injecting.crack = values["crack"]
        faults_injecting.dirt = values["dirt"]
        faults_injecting.ice = values["ice"]
        faults_injecting.rain = values["rain"]
        faults_injecting.blur = values["blur"]
        faults_injecting.brightness = values["brightness"]
        faults_injecting.no_chrom_ac = values["no_chrom_ac"]
        faults_injecting.dead_pixels = values["dead_pixels"]
        faults_injecting.dead_pixels_line = values["dead_pixels_line"]
        faults_injecting.no_bayer_filter = values["no_bayer_filter"]
        faults_injecting.no_Demosaicing = values["no_Demosaicing"]
        faults_injecting.speckle_noise = values["speckle_noise"]
        faults_injecting.sharpness = values["sharpness"]
        faults_injecting.darkness = values["darkness"]
        faults_injecting.iter_null()
        faults_injecting.update_fault_activation_array()

        strength_list.clear()
        if values["fault_strength1"]:
            strength_list.append(1)
        if values["fault_strength2"]:
            strength_list.append(2)
        if values["fault_strength3"]:
            strength_list.append(3)
        if values["fault_strength4"]:
            strength_list.append(4)
        if values["fault_strength5"]:
            strength_list.append(5)
        output_path = values['set_outputpath']
        if not os.path.exists(output_path + output_folder):
            os.makedirs(output_path + output_folder)
        source_path = values['set_sourcepath']

        faults_injecting.generate_fault_message()  # generate for strength also
        for file in glob.iglob(source_path + "//*.jpg", recursive=True):
            image = Image.open(file).convert('RGB')
            image = image.resize(output_size)
            faults_injecting.faults_generating(image, strength_list, output_path + output_folder)
        print(source_path, output_path + '\output')
        subprocess.Popen(r'explorer \open,' + output_path + '\output')

if __name__ == '__main__':
    pass
