import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/arommel/Downloads/examen_e1_ws/install/visual_pubsub'
