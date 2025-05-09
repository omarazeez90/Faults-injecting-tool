# fault injecting tool
The tool generate different RGB image from each image sample and saved it in a specific path defined by the user.

# requirements.txt
-Python version: 3.7.9<br>
-PySimpleGUI==4.60.4<br>
-Pillow==9.4.0<br>
-opencv_python==4.8.0.74<br>
-numpy==1.21.6<br>
-splitfolders==0.5.1<br>
## To-Do:
-Add snow effect that covers the ground.<br>
-Add different implementation to rain without overlaying.<br>
-Add different implementation to fog without overlaying.<br>
-Add blur area instead of blurring the whole image.<br>
-Add shadow on the ground fault.<br>
-Create better overlay fault samples.<br>
-Simplifying the code in later stage for more understanding.<br>

# Description to the files 
### Fault.py
contain the fault class and the implementation of the faults methods.

### main.py
contain a simple graphical user interface for easy use.
### Lena.jpg
it is an RGB image used as test sample.
### output
the tool create the output folder if it does not exist in the specified path and saves the generated samples in it in a different folder depending on the selected fault.
### split_data.py
used to split dataset to (train, test, and val) with specified ratio.
# Graphical user interface
<p align="center">
<img src="https://github.com/omarMohammed-USI/omarMohammed-USI/blob/main/faults_GUI.png" height=400>
</p>

# Test sample
<p align="center">
<img src="https://github.com/omarMohammed-USI/omarMohammed-USI/blob/main/Lena.jpg" height=400>
</p>

# Result
the faults shown are blur, crack, speckle noise and dead pixels for the strength from 1 to 5
<p align="center">
<img src="https://github.com/omarMohammed-USI/omarMohammed-USI/blob/main/fault%20samples.jpg" height=720>
</p>

# Contact data
[Omar Mohammed](https://www.eti.uni-siegen.de/mt/mitarbeiter/?lang=de) | University of Siegen
