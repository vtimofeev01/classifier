from openvino.inference_engine.ie_api import IECore
import os

xml = '/home/imt/work/pyqt5-images-classifier/dataset/save/FP16/checkpoint-best.xml'
ie = IECore()
n = ie.read_network(model=xml, weights=os.path.splitext(xml)[0] + '.bin')
z = ie.load_network(n, device_name='CPU')
it = next(iter(z.inputs))
print(it, z.inputs[it].shape)
ot = next(iter(z.outputs))
print(ot, z.outputs[ot].shape)



