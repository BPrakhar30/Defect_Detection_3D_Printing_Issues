# Defect_Detection_3D_Printing_Issues

Goal of the project - 

3D printing is an innovative way to fabricate parts. However, 3D printing is also known to be prone errors. Most of these errors are visible via a close-up camera mounted right near the printer nozzle.

The goal is to predict 1 specific kind of error - under extrusion.

A little background on 3D printing - 
In a nutshell, 3D printing is a process of pushing molten plastic through a small nozzle onto a print bed. 

What does a good 3D print looks like?
When 3D printing goes well, the plastic will form smooth, continuous, and neat geometrical shapes.

What does under extrusion looks like?
When under extrusion happens, the plastic will have all kinds of artifacts, such as wrinkles, bubbles, curls, etc. 

There are plenty of pictures in the training and test dataset that are labelled as "under extrusion" but look very much like good prints. Welcome to the messy real world. ;)

Prior researches - 
This project is inspired by this paper: https://www.nature.com/articles/s41467-022-31985-y

The source code for this paper is: https://github.com/cam-cambridge/caxton

However, a word of caution: this research is based on a very pure lab environment. It uses data from the printers that share the same setup and hence very close to each other. Our dataset is more diverse (from 7 printers that look distinct from each other).
