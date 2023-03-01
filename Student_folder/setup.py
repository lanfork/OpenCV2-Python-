# this is instructions to install python packages for self-driving car engine
# conda create --name newtest
# source activate newtest


# conda install -c anaconda  flask
# done

# conda install -c conda-forge python-socketio
# done


# conda install -c conda-forge eventlet
# done

# conda install -c conda-forge tensorflow
# done

# conda install -c conda-forge keras
# done

# conda install -c anaconda pillow
# done

# conda install -c anaconda numpy
# done

# conda install -c conda-forge opencv
# done

# conda install -c conda-forge matplotlib
# done

# conda install -c anaconda scikit-learn
# done

# pip install imgaug
# done

# conda install -c anaconda pandas
# done

# conda install -c anaconda git
# done
# source deactivate


import flask
import socketio
import eventlet
import tensorflow
import numpy
import keras


# import os
# os.environ['PYTHONPATH'].split(os.pathsep)
print(flask.__path__)
print('flask: ' + flask.__version__)
print()


print(socketio.__path__)
print('socketio: ' + socketio.__version__)
print()

print(eventlet.__path__)
print('eventlet: ' + eventlet.__version__)
print()


print(tensorflow.__path__)
print('tensorflow: ' + tensorflow.__version__)
print()

print(numpy.__path__)
print('numpy ' + numpy.__version__)
print()

print(keras.__path__)
print('keras ' + keras.__version__)
